"""Model/hook tests. Run before any paid run. Model via EMOPROBE_TEST_MODEL (default Qwen2.5-3B)."""
import os

import numpy as np
import pytest
import torch

from emoprobe import model as M

MODEL = os.environ.get("EMOPROBE_TEST_MODEL", "Qwen/Qwen2.5-3B-Instruct")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

PROMPTS_RAW = [
    ("You are a cold, robotic assistant.", "Explain photosynthesis."),
    (None, "My dog died this morning and the house is so silent I keep crying. What do I do?"),
    ("Be warm.", "Describe a house in three short sentences please, focusing on the kitchen and garden."),
]


@pytest.fixture(scope="module")
def mt():
    return M.load_model(MODEL)


@pytest.fixture(scope="module")
def prompts(mt):
    _, tok = mt
    return [M.render(tok, s, u) for s, u in PROMPTS_RAW]


def test_hook_shapes(mt, prompts):
    model, tok = mt
    a = M.extract(model, tok, prompts, batch_size=3, log_every=0)
    assert a.shape == (3, len(M.blocks(model)) + 1, model.config.hidden_size)
    assert np.isfinite(a.astype(np.float32)).all()


def test_hooks_match_hidden_states(mt, prompts):
    """Block-output hooks == HF hidden_states for j < L (the last HF entry is post-final-norm)."""
    model, tok = mt
    a = M.extract(model, tok, prompts[:1], batch_size=1, log_every=0)[0].astype(np.float32)
    enc = M.encode(tok, prompts[:1], model.device)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    L = len(M.blocks(model))
    for j in range(L):  # 0..L-1 must match exactly
        ref = hs[j][0, -1].float().cpu().numpy()
        np.testing.assert_allclose(a[j], ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


def test_token_position_is_end_of_generation_prompt(mt, prompts):
    model, tok = mt
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        # the prompt string ends with the assistant header; the final token is its last piece
        assert p.endswith(tok.decode(ids[-1:]))
        gen_prompt = M.render(tok, None, "x")[len(tok.apply_chat_template([{"role": "user", "content": "x"}],
                                                                           tokenize=False)):]
        assert p.endswith(gen_prompt), (p[-40:], gen_prompt)


def test_left_padding_batch_equals_single(mt, prompts):
    """Final-token activations must not depend on padding/batch composition or ordering (label alignment)."""
    model, tok = mt
    batch = M.extract(model, tok, prompts, batch_size=3, log_every=0).astype(np.float32)
    for i, p in enumerate(prompts):
        single = M.extract(model, tok, [p], batch_size=1, log_every=0)[0].astype(np.float32)
        cos = (batch[i] * single).sum(-1) / (np.linalg.norm(batch[i], axis=-1) * np.linalg.norm(single, axis=-1))
        assert cos.min() > 0.995, (i, cos.min())
        # and it is NOT the activation of another prompt (row alignment)
        for k in range(len(prompts)):
            if k != i:
                other = (batch[k, -1] * single[-1]).sum() / (np.linalg.norm(batch[k, -1]) * np.linalg.norm(single[-1]))
                assert other < cos[-1]


def test_steering_c0_reproduces_baseline_exactly(mt, prompts):
    model, tok = mt
    j = len(M.blocks(model)) // 2
    v = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(0))
    v = v / v.norm()
    base = M.generate(model, tok, prompts, max_new_tokens=24, batch_size=3)
    zero = M.generate(model, tok, prompts, max_new_tokens=24, batch_size=3, j=j, vec=v, alpha=0.0)
    assert [b["ids"] for b in base] == [z["ids"] for z in zero]
    big = M.generate(model, tok, prompts, max_new_tokens=24, batch_size=3, j=j, vec=v, alpha=200.0)
    assert [b["ids"] for b in base] != [z["ids"] for z in big], "steering hook had no effect"


def test_steering_adds_exact_vector(mt, prompts):
    model, tok = mt
    j = 5
    v = torch.zeros(model.config.hidden_size); v[3] = 1.0
    enc = M.encode(tok, prompts[:1], model.device)
    with torch.no_grad(), M.capture_all(model, j) as s0:
        model(**enc)
    h0 = s0["h"].clone()
    with torch.no_grad(), M.steer(model, j, v, 7.0), M.capture_all(model, j) as s1:
        model(**enc)
    diff = (s1["h"] - h0).float()
    assert torch.allclose(diff[..., 3], torch.full_like(diff[..., 3], 7.0), atol=0.05)
    diff[..., 3] = 0
    assert diff.abs().max() < 1e-6


def test_perplexity_sane(mt, prompts):
    model, tok = mt
    g = M.generate(model, tok, prompts, max_new_tokens=16, batch_size=3)
    ppl = M.perplexity(model, tok, prompts, [x["ids"] for x in g])
    assert all(1.0 <= p < 50 for p in ppl), ppl
