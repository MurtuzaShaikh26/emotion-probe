"""Judge every generation with the frozen rubric (Fireworks gpt-oss-120b, T=0), blind and in shuffled order.

Resumable (results/<slug>/judgments.jsonl keyed by gen_id). Hard budget guard across all models
(results/judge_costs.json). Usage: python scripts/run_judge.py --model unsloth/Meta-Llama-3.1-8B-Instruct
"""
import argparse, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv
from emoprobe import judge as Jg
from emoprobe.datasets import load_jsonl
from emoprobe.paths import ROOT, RESULTS, slug

load_dotenv(ROOT / ".env")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--cap-usd", type=float, default=7.5)  # brief: hard cap $8 on Fireworks
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="judge only the first N (shuffled) items; 0 = all")
a = ap.parse_args()
S = slug(a.model)

gens = load_jsonl(RESULTS / S / "generations.jsonl")
out_path = RESULTS / S / "judgments.jsonl"
done = {r["gen_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
todo = [g for g in gens if g["gen_id"] not in done]
random.Random(0).shuffle(todo)  # order randomised; the judge sees only request + response
if a.limit:
    todo = todo[:a.limit]

cost_path = RESULTS / "judge_costs.json"
costs = json.loads(cost_path.read_text()) if cost_path.exists() else {"prompt_tokens": 0, "completion_tokens": 0}
lock = threading.Lock()


def spent():
    return costs["prompt_tokens"] / 1e6 * Jg.PRICE_IN + costs["completion_tokens"] / 1e6 * Jg.PRICE_OUT


print(f"{len(todo)} to judge; spent so far ${spent():.3f} (cap ${a.cap_usd})", flush=True)
cl = Jg.client()
stop = threading.Event()


def work(g):
    if stop.is_set():
        return None
    res, pt, ct, raw = Jg.judge_one(cl, g["request"], g["text"])
    return g, res, pt, ct, raw


n_ok = n_fail = 0
with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as pool:
    futs = [pool.submit(work, g) for g in todo]
    for k, f in enumerate(as_completed(futs)):
        r = f.result()
        if r is None:
            continue
        g, res, pt, ct, raw = r
        with lock:
            costs["prompt_tokens"] += pt; costs["completion_tokens"] += ct
            if res is None:
                n_fail += 1
            else:
                n_ok += 1
                fh.write(json.dumps(dict(gen_id=g["gen_id"], judge=Jg.JUDGE_MODEL, **res, raw=raw[-400:])) + "\n")
                fh.flush()
            if spent() > a.cap_usd:
                stop.set()
                print("BUDGET CAP REACHED — stopping", flush=True)
            if k % 100 == 0:
                cost_path.write_text(json.dumps(costs))
                print(f"  {k}/{len(todo)} ok={n_ok} fail={n_fail} spent=${spent():.3f}", flush=True)
cost_path.write_text(json.dumps(costs))
print(f"done: ok={n_ok} fail={n_fail}; total judge spend ${spent():.3f}")
