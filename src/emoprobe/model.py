"""Model loading, prompt rendering, activation caching and steering with plain HF forward hooks.

Layer indexing convention (used everywhere in this project)
-----------------------------------------------------------
Residual index j = 0 is the embedding output; j = i + 1 is the OUTPUT of decoder block i.
So a model with L blocks gives L + 1 residual snapshots. Steering "at j" adds to the output of block j - 1,
which is exactly the residual stream that block j reads. (We do not use `output_hidden_states` because in
HF the last entry has the final RMSNorm applied, which would make the last layer inconsistent.)
"""
from __future__ import annotations

import contextlib
import math
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str, device: str = "cuda", dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def blocks(model):
    return model.model.layers


def n_resid(model) -> int:
    return len(blocks(model)) + 1


def render(tok, system: str | None, user: str, fmt: str = "chat") -> str:
    """Return the full prompt string (special tokens included); encode with add_special_tokens=False."""
    if fmt == "chat":
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if fmt == "plain":
        bos = tok.bos_token or ""
        head = f"{system}\n\n" if system else ""
        return f"{bos}{head}User: {user}\nAssistant:"
    raise ValueError(fmt)


def encode(tok, prompts: list[str], device):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


def _pos_ids(mask: torch.Tensor) -> torch.Tensor:
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


@contextlib.contextmanager
def capture_last_token(model, store: list):
    """Hook every block; append [batch, L+1, d] residuals at the final position (left padding => -1)."""
    acts: dict[int, torch.Tensor] = {}
    handles = []

    def mk(i):
        def hook(mod, inp, out):
            acts[i + 1] = _out_tensor(out)[:, -1, :].detach()
        return hook

    def emb_hook(mod, inp, out):
        acts[0] = out[:, -1, :].detach()

    handles.append(model.model.embed_tokens.register_forward_hook(emb_hook))
    for i, b in enumerate(blocks(model)):
        handles.append(b.register_forward_hook(mk(i)))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def extract(model, tok, prompts: list[str], batch_size: int = 16, log_every: int = 20) -> np.ndarray:
    """Residual stream at the final prompt token, all layers. Returns float16 array [N, L+1, d]."""
    dev = next(model.parameters()).device
    out = []
    order = np.argsort([-len(p) for p in prompts])  # length-sort to reduce padding
    for bi, s in enumerate(range(0, len(prompts), batch_size)):
        idx = order[s:s + batch_size]
        enc = encode(tok, [prompts[i] for i in idx], dev)
        with capture_last_token(model, []) as acts:
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                  position_ids=_pos_ids(enc["attention_mask"]), use_cache=False)
        stacked = torch.stack([acts[j] for j in range(n_resid(model))], dim=1)
        out.append((idx, stacked.float().cpu().numpy().astype(np.float16)))
        if log_every and bi % log_every == 0:
            print(f"  extract batch {bi} / {math.ceil(len(prompts) / batch_size)}", flush=True)
    res = np.zeros((len(prompts),) + out[0][1].shape[1:], dtype=np.float16)
    for idx, a in out:
        res[idx] = a
    return res


@torch.no_grad()
def token_norms(model, tok, prompts: list[str], j: int) -> np.ndarray:
    """Per-token residual norms at residual index j over all non-pad, non-first tokens."""
    dev = next(model.parameters()).device
    enc = encode(tok, prompts, dev)
    with capture_all(model, j) as store:
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
              position_ids=_pos_ids(enc["attention_mask"]), use_cache=False)
    h = store["h"].float()
    mask = enc["attention_mask"].bool().clone()
    first = mask.int().argmax(-1)
    mask[torch.arange(mask.shape[0]), first] = False  # drop BOS/first token (often a huge-norm "sink")
    return h.norm(dim=-1)[mask].cpu().numpy()


@contextlib.contextmanager
def capture_all(model, j: int):
    store = {}
    if j == 0:
        mod = model.model.embed_tokens
        h = mod.register_forward_hook(lambda m, i, o: store.__setitem__("h", o.detach()))
    else:
        mod = blocks(model)[j - 1]
        h = mod.register_forward_hook(lambda m, i, o: store.__setitem__("h", _out_tensor(o).detach()))
    try:
        yield store
    finally:
        h.remove()


@contextlib.contextmanager
def steer(model, j: int, vec: torch.Tensor | None, alpha: float):
    """Add alpha * vec to the residual stream at index j (j >= 1), at ALL positions (prompt + generated)."""
    if vec is None:
        yield
        return
    assert j >= 1
    v = vec.to(next(model.parameters()).device, next(model.parameters()).dtype)
    add = alpha * v

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + add,) + tuple(out[1:])
        return out + add

    h = blocks(model)[j - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 128, batch_size: int = 16,
             j: int | None = None, vec: torch.Tensor | None = None, alpha: float = 0.0) -> list[dict]:
    """Greedy decoding. Returns list of {text, ids} for each prompt (ids = generated ids, cut at EOS)."""
    dev = next(model.parameters()).device
    eos = set([tok.eos_token_id] + ([model.generation_config.eos_token_id]
                                    if isinstance(model.generation_config.eos_token_id, int)
                                    else list(model.generation_config.eos_token_id or [])))
    res = [None] * len(prompts)
    order = np.argsort([-len(p) for p in prompts])
    for s in range(0, len(prompts), batch_size):
        idx = order[s:s + batch_size]
        enc = encode(tok, [prompts[i] for i in idx], dev)
        ctx = steer(model, j, vec, alpha) if vec is not None else contextlib.nullcontext()
        with ctx:
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:].cpu().tolist()
        for i, g in zip(idx, gen):
            cut = next((k + 1 for k, t in enumerate(g) if t in eos), len(g))
            ids = g[:cut]
            res[i] = dict(ids=ids, text=tok.decode(ids, skip_special_tokens=True).strip())
    return res


@torch.no_grad()
def perplexity(model, tok, prompts: list[str], gen_ids: list[list[int]], batch_size: int = 8) -> list[float]:
    """PPL of each generated continuation given its prompt, under the (unsteered) model."""
    dev = next(model.parameters()).device
    res = [float("nan")] * len(prompts)
    for s in range(0, len(prompts), batch_size):
        chunk = list(range(s, min(s + batch_size, len(prompts))))
        seqs, n_prompt = [], []
        for i in chunk:
            p = tok(prompts[i], add_special_tokens=False)["input_ids"]
            seqs.append(p + gen_ids[i]); n_prompt.append(len(p))
        T = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, T - len(x):] = torch.tensor(x); mask[r, T - len(x):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=_pos_ids(mask), use_cache=False).logits
        logp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
        for r, i in enumerate(chunk):
            n_gen = len(gen_ids[i])
            if n_gen == 0:
                continue
            lp = logp[r, -n_gen:]
            res[i] = float(torch.exp(-lp.mean()).item())
    return res


def distinct2(text: str) -> float:
    w = re.findall(r"\w+", text.lower())
    if len(w) < 3:
        return 1.0
    bi = list(zip(w, w[1:]))
    return len(set(bi)) / len(bi)
