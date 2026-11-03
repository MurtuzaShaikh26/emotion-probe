"""Combine generations + judgments -> results/<slug>/steering_metrics.json (dose-response, contrasts, H3/H4).

Usage: python scripts/analyze_steering.py --model unsloth/Meta-Llama-3.1-8B-Instruct
"""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import spearmanr
from emoprobe.datasets import load_jsonl
from emoprobe.paths import RESULTS, slug

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
a = ap.parse_args()
S = slug(a.model)
gens = load_jsonl(RESULTS / S / "generations.jsonl")
jp = RESULTS / S / "judgments.jsonl"
judg = {r["gen_id"]: r for r in load_jsonl(jp)} if jp.exists() else {}
VECS, CS = ["d1", "d2", "d1_deconf", "random"], [-8, -4, -2, 0, 2, 4, 8]
prompts = sorted({g["prompt_id"] for g in gens})
P_IDX = {p: i for i, p in enumerate(prompts)}
B = np.random.default_rng(0).integers(0, len(prompts), (2000, len(prompts)))


def table(field, fn=None):
    """[vector][c] -> array over prompts (nan where missing). c=0 shared baseline."""
    fn = fn or (lambda g, j: g.get(field))
    out = {}
    for v in VECS:
        out[v] = {}
        for c in CS:
            arr = np.full(len(prompts), np.nan)
            for g in gens:
                if (g["vector"] == v and g["c"] == c) or (c == 0 and g["vector"] == "none"):
                    val = fn(g, judg.get(g["gen_id"]))
                    if val is not None:
                        arr[P_IDX[g["prompt_id"]]] = float(val)
            out[v][c] = arr
    return out


def ci(arr):
    m = ~np.isnan(arr)
    if m.sum() == 0:
        return [None, None, None]
    boots = np.nanmean(arr[B], axis=1)
    return [float(np.nanmean(arr)), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]


T = dict(
    emotion=table("emotion", lambda g, j: j["emotion"] if j else None),
    task_completed=table("tc", lambda g, j: j["task_completed"] if j else None),
    judge_refusal=table("jr", lambda g, j: j["refusal"] if j else None),
    log_ppl=table("ppl", lambda g, j: np.log(g["ppl"]) if g["ppl"] == g["ppl"] else None),
    distinct2=table("distinct2"), n_tokens=table("n_tokens"),
    rx_refusal=table("rx_refusal"), rx_apology=table("rx_apology"), rx_bullets=table("rx_bullets"),
)
dose = {k: {v: {str(c): ci(T[k][v][c]) for c in CS} for v in VECS} for k in T}
# geometric-mean perplexity CI = exp of log-ppl CI
dose["ppl_geomean"] = {v: {c: [float(np.exp(x)) if x is not None else None for x in dose["log_ppl"][v][c]]
                           for c in dose["log_ppl"][v]} for v in VECS}

contrasts = {}
for c in CS:
    if c == 0:
        continue
    for a_, b_ in (("d1", "random"), ("d2", "random"), ("d1_deconf", "random"), ("d2", "d1"), ("d1_deconf", "d1")):
        contrasts[f"{a_}-{b_}@{c:+d}"] = ci(T["emotion"][a_][c] - T["emotion"][b_][c])

n_judged = sum(g["gen_id"] in judg for g in gens)
res = dict(model=a.model, n_generations=len(gens), n_judged=n_judged, n_prompts=len(prompts),
           layer=gens[0]["layer"], alpha_per_c=abs(next(g["alpha"] / g["c"] for g in gens if g["c"])),
           dose=dose, contrasts_emotion=contrasts)

# pre-registered verdicts (c = +4)
if n_judged:
    h3_diff = contrasts["d1-random@+4"]
    tc4 = dose["task_completed"]["d1"]["4"][0]
    res["H3"] = dict(diff_d1_minus_random=h3_diff, task_completion_d1=tc4,
                     supported=bool(h3_diff[1] > 0 and tc4 >= 0.70))
    d21, d2r = contrasts["d2-d1@+4"], contrasts["d2-random@+4"]
    res["H4"] = dict(diff_d2_minus_d1=d21, diff_d2_minus_random=d2r,
                     supported=bool(d21[1] > -1.0 and d2r[1] > 0))

# judge vs second-rater agreement: Murt's hand labels (human) and/or Claude Opus 5.5's blind labels (an LLM, not human)
key = RESULTS / "human_labels_key.json"
for rater, hl in (("human", RESULTS / "human_labels.csv"), ("claude", RESULTS / "claude_labels.csv")):
    if not (hl.exists() and key.exists()):
        continue
    kmap = json.loads(key.read_text())
    allj = {}
    for s_ in ("llama8b", "qwen3b"):
        p_ = RESULTS / s_ / "judgments.jsonl"
        if p_.exists():
            allj.update({r["gen_id"]: r for r in load_jsonl(p_)})
    he, je, ht, jt = [], [], [], []
    with open(hl, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            e, t = row.get("emotional_intensity_1to10", "").strip(), row.get("task_completed_yes_no", "").strip().lower()
            gid = kmap.get(row["label_id"])
            if not e or gid not in allj:
                continue
            he.append(float(e)); je.append(allj[gid]["emotion"])
            if t[:1] in ("y", "n"):
                ht.append(t[:1] == "y"); jt.append(bool(allj[gid]["task_completed"]))
    if len(he) >= 5:
        he, je = np.array(he), np.array(je)
        agree = dict(rater=rater, n=len(he), spearman=float(spearmanr(he, je).correlation),
                     mean_abs_diff=float(np.abs(he - je).mean()), within1=float((np.abs(he - je) <= 1).mean()),
                     judge_minus_human_mean=float((je - he).mean()), rater_sd=float(he.std()), judge_sd=float(je.std()))
        if ht:
            ht, jt = np.array(ht), np.array(jt)
            po = float((ht == jt).mean())
            pe = float(ht.mean() * jt.mean() + (1 - ht.mean()) * (1 - jt.mean()))
            agree.update(task_n=len(ht), task_agreement=po, task_kappa=(po - pe) / (1 - pe) if pe < 1 else float("nan"))
        (RESULTS / f"judge_{rater}_agreement.json").write_text(json.dumps(agree, indent=1))
        res[f"judge_{rater}_agreement"] = agree

(RESULTS / S / "steering_metrics.json").write_text(json.dumps(res, indent=1))
print(f"{S}: {len(gens)} generations, {n_judged} judged")
for v in VECS:
    print(f"  {v:9s} emotion " + " ".join(f"{c:+d}:{dose['emotion'][v][str(c)][0] if dose['emotion'][v][str(c)][0] is not None else float('nan'):.2f}" for c in CS))
for v in VECS:
    print(f"  {v:9s} ppl     " + " ".join(f"{c:+d}:{dose['ppl_geomean'][v][str(c)][0]:.2f}" for c in CS))
for k in ("H3", "H4"):
    if k in res:
        print(k, res[k])
