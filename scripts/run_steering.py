"""Stage 2 (control): scale calibration, steered generation, perplexity, regex side effects.

Needs cache/<slug>/directions.npz from run_probes.py. Resumable: each condition is appended to
results/<slug>/generations.jsonl as it finishes and skipped on restart.
Usage: python scripts/run_steering.py --model Qwen/Qwen2.5-3B-Instruct [--smoke]
"""
import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from emoprobe import model as M
from emoprobe.datasets import load_jsonl
from emoprobe.paths import DATA, CACHE, RESULTS, slug

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--batch", type=int, default=30)
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--smoke", action="store_true", help="4 prompts, c in {-8,0,8}, 32 tokens; writes to *_smoke files")
a = ap.parse_args()
S = slug(a.model)
OUT = RESULTS / S; OUT.mkdir(parents=True, exist_ok=True)
sfx = "_smoke" if a.smoke else ""
t0 = time.time()

FRACS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
C_GRID = [-8, -4, -2, 2, 4, 8]
VECTORS = ["d1", "d2", "d1_deconf", "random"]

ex = load_jsonl(DATA / "examples.jsonl")
steer_ex = [e for e in ex if e["dataset"] == "steer"]
calib_ex = [e for e in ex if e["dataset"] == "calib"]
if a.smoke:
    steer_ex, C_GRID, a.max_new = steer_ex[:4], [-8, 8], 32

dz = np.load(CACHE / S / "directions.npz")
j = int(dz["steer_layer"])
model, tok = M.load_model(a.model)
d = model.config.hidden_size
g = torch.Generator().manual_seed(0)
vecs = {k: torch.tensor(dz[k][j], dtype=torch.float32) for k in ("d1", "d2", "d1_deconf")}
vecs["random"] = torch.randn(d, generator=g)
vecs = {k: v / v.norm() for k, v in vecs.items()}
print(f"layer {j}; cos(d1,d2)={float(vecs['d1'] @ vecs['d2']):+.3f} cos(d1,deconf)={float(vecs['d1'] @ vecs['d1_deconf']):+.3f} "
      f"cos(d1,random)={float(vecs['d1'] @ vecs['random']):+.3f}", flush=True)

steer_prompts = [M.render(tok, None, e["user"]) for e in steer_ex]
calib_prompts = [M.render(tok, None, e["user"]) for e in calib_ex]

# ---------------- calibration (fluency only) ----------------
cal_path = OUT / f"calibration{sfx}.json"
if cal_path.exists():
    cal = json.loads(cal_path.read_text())
else:
    norms = M.token_norms(model, tok, calib_prompts, j)
    nu = float(np.median(norms))
    def fluency(alpha):
        outs = M.generate(model, tok, calib_prompts, max_new_tokens=64, batch_size=len(calib_prompts), j=j,
                          vec=vecs["d1"] if alpha else None, alpha=alpha)
        ppl = M.perplexity(model, tok, calib_prompts, [o["ids"] for o in outs])
        return float(np.nanmean(ppl)), float(np.mean([M.distinct2(o["text"]) for o in outs])), outs[0]["text"][:200]
    base_ppl, base_d2, base_txt = fluency(0.0)
    rows, f_star = [], None
    for f in FRACS:
        r = dict(frac=f)
        for sign in (+1, -1):
            ppl, d2_, txt = fluency(sign * f * nu)
            r[f"ppl_{sign:+d}"], r[f"distinct2_{sign:+d}"], r[f"example_{sign:+d}"] = ppl, d2_, txt
        broken = max(r["ppl_+1"], r["ppl_-1"]) > 3 * base_ppl or min(r["distinct2_+1"], r["distinct2_-1"]) < 0.5
        r["broken"] = bool(broken)
        rows.append(r)
        print(f"  calib f={f:.2f}: ppl +{r['ppl_+1']:.1f} / -{r['ppl_-1']:.1f} (base {base_ppl:.2f}), "
              f"distinct2 {r['distinct2_+1']:.2f}/{r['distinct2_-1']:.2f} broken={broken}", flush=True)
        if broken and f_star is None:
            f_star = f
            break  # rule: smallest breaking f
    if f_star is None:
        f_star = 1.0
    cal = dict(layer=j, typical_norm=nu, base_ppl=base_ppl, base_distinct2=base_d2, base_example=base_txt,
               rows=rows, f_star=f_star, unit=f_star * nu / 8, rule="c=8 <-> f* x typical norm (PREREG)")
    cal_path.write_text(json.dumps(cal, indent=1))
unit = cal["unit"]
print(f"calibration: typical norm {cal['typical_norm']:.1f}, f*={cal['f_star']}, alpha per unit c = {unit:.3f} "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---------------- generation ----------------
gen_path = OUT / f"generations{sfx}.jsonl"
done = set()
if gen_path.exists():
    for r in load_jsonl(gen_path):
        done.add((r["vector"], r["c"]))
conds = [("none", 0)] + [(v, c) for v in VECTORS for c in C_GRID]
REFUSAL = re.compile(r"\b(I can(?:not|'t)|I won't|I'm (?:not able|unable)|I am (?:not able|unable)|As an AI)\b", re.I)
APOLOGY = re.compile(r"\b(sorry|apologi[sz]e|apologies)\b", re.I)


def bullets(t):
    return sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s+", line)) for line in t.splitlines()) >= 2


with open(gen_path, "a", encoding="utf-8") as fh:
    for vname, c in conds:
        if (vname, c) in done:
            continue
        tc = time.time()
        alpha = c * unit
        outs = M.generate(model, tok, steer_prompts, max_new_tokens=a.max_new, batch_size=a.batch, j=j,
                          vec=None if vname == "none" else vecs[vname], alpha=alpha)
        ppl = M.perplexity(model, tok, steer_prompts, [o["ids"] for o in outs])
        for e, p, o, pp in zip(steer_ex, steer_prompts, outs, ppl):
            fh.write(json.dumps(dict(
                gen_id=f"{S}|{vname}|{c}|{e['id']}", model=a.model, prompt_id=e["id"], request=e["user"],
                vector=vname, c=c, alpha=alpha, layer=j, text=o["text"], n_tokens=len(o["ids"]), ppl=pp,
                distinct2=M.distinct2(o["text"]), rx_refusal=bool(REFUSAL.search(o["text"])),
                rx_apology=bool(APOLOGY.search(o["text"])), rx_bullets=bullets(o["text"])), ensure_ascii=False) + "\n")
        fh.flush()
        print(f"  {vname:9s} c={c:+d}: mean ppl {np.nanmean(ppl):.2f}  ({time.time()-tc:.0f}s)  "
              f"ex: {outs[0]['text'][:90]!r}", flush=True)
print(f"done in {time.time()-t0:.0f}s")
