"""Report D2 pair length matching in words and tokens (Llama + Qwen tokenizers), incl. length-only AUROC."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from transformers import AutoTokenizer
from emoprobe.texts import D2_PAIRS
from emoprobe.datasets import load_jsonl
from emoprobe.probes import auroc
ex = [e for e in load_jsonl(Path(__file__).resolve().parents[1] / "data/examples.jsonl") if e["dataset"] == "d2"]
wd = np.array([len(a.split()) - len(b.split()) for ps in D2_PAIRS.values() for _, a, b in ps])
print(f"words: mean diff {wd.mean():+.2f}, max |diff| {abs(wd).max()}")
for m in ["unsloth/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]:
    t = AutoTokenizer.from_pretrained(m)
    n = lambda s: len(t(s, add_special_tokens=False)["input_ids"])
    d = []
    for tp, ps in D2_PAIRS.items():
        for j, (_, a, b) in enumerate(ps):
            d.append(n(a) - n(b))
            if abs(d[-1]) >= 2 and "Llama" in m: print("  ", f"{tp}:{j}", d[-1])
    d = np.array(d)
    au = {sp: round(auroc([n(e["user"]) for e in ex if e["split"] == sp], [e["label"] for e in ex if e["split"] == sp]), 3)
          for sp in ("train", "test")}
    print(f"{m}: token diff mean {d.mean():+.2f}, max |diff| {abs(d).max()}, token-length AUROC {au}")
