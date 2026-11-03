"""Make results/human_labels.csv: 40 Llama outputs for blind hand-labelling (judge validation).

Stratified, fixed seed: 1 output from each of the 24 (vector, c != 0) cells + 6 unsteered + 10 random extra.
The condition is hidden; the mapping lives in results/human_labels_key.json (don't open it while labelling).
"""
import csv, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emoprobe.datasets import load_jsonl
from emoprobe.paths import RESULTS

out = RESULTS / "human_labels.csv"
if out.exists():
    sys.exit(f"{out} exists — refusing to overwrite (it may contain your labels)")
rng = random.Random(0)
gens = load_jsonl(RESULTS / "llama8b" / "generations.jsonl")
cells = {}
for g in gens:
    cells.setdefault((g["vector"], g["c"]), []).append(g)
pick = []
for key in sorted(k for k in cells if k[0] != "none"):
    pick.append(rng.choice(cells[key]))
pick += rng.sample(cells[("none", 0)], 6)
rest = [g for g in gens if g not in pick]
pick += rng.sample(rest, 40 - len(pick))
rng.shuffle(pick)
key = {}
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["label_id", "request", "response", "emotional_intensity_1to10", "task_completed_yes_no"])
    for i, g in enumerate(pick):
        lid = f"H{i+1:02d}"
        key[lid] = g["gen_id"]
        w.writerow([lid, g["request"], g["text"], "", ""])
(RESULTS / "human_labels_key.json").write_text(json.dumps(key, indent=1))
print(f"wrote {out} ({len(pick)} rows)")
