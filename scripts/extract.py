"""Extract final-prompt-token residuals at every layer for all labelled examples.

Writes cache/<slug>/acts.npy  [N, L+1, d] float16  and  cache/<slug>/ids.json (row order).
Usage: python scripts/extract.py --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from emoprobe import model as M
from emoprobe.datasets import load_jsonl
from emoprobe.paths import DATA, CACHE, slug

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

ex = [e for e in load_jsonl(DATA / "examples.jsonl") if e["dataset"] not in ("steer", "calib")]
out = CACHE / slug(a.model); out.mkdir(parents=True, exist_ok=True)
t = time.time()
model, tok = M.load_model(a.model)
prompts = [M.render(tok, e["system"], e["user"], e["fmt"]) for e in ex]
acts = M.extract(model, tok, prompts, batch_size=a.batch)
np.save(out / "acts.npy", acts)
(out / "ids.json").write_text(json.dumps([e["id"] for e in ex]))
(out / "prompt_examples.json").write_text(json.dumps({e["dataset"]: p for e, p in zip(ex, prompts)}, indent=1))
print(f"saved {acts.shape} in {time.time()-t:.0f}s")
