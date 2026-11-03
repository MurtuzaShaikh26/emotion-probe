"""Build data/examples.jsonl, run leakage + label checks, write results/dataset_stats.json."""
import collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emoprobe.datasets import build_all, check_leakage, label_alignment_ok, d2_stats, save_jsonl

ex = build_all()
leak = check_leakage(ex)
label_alignment_ok(ex)
stats = dict(counts=dict(collections.Counter((e["dataset"], e["split"]) for e in ex).most_common()),
             leakage=leak, d2=d2_stats(ex))
stats["counts"] = {f"{k[0]}/{k[1]}": v for k, v in stats["counts"].items()}
save_jsonl(ex, ROOT / "data/examples.jsonl")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/dataset_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
print("OK: no leakage, labels aligned, n =", len(ex))
