"""Stage 1 (reading): all probe analyses for one model.

Reads cache/<slug>/acts.npy, writes results/<slug>/probe_metrics.json and cache/<slug>/directions.npz.
Usage: python scripts/run_probes.py --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from emoprobe import probes as P
from emoprobe.datasets import load_jsonl
from emoprobe.paths import DATA, CACHE, RESULTS, slug

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--n-perm", type=int, default=1000)
a = ap.parse_args()
S = slug(a.model)
t0 = time.time()

ex_all = {e["id"]: e for e in load_jsonl(DATA / "examples.jsonl")}
ids = json.loads((CACHE / S / "ids.json").read_text())
ex = [ex_all[i] for i in ids]
acts = np.load(CACHE / S / "acts.npy", mmap_mode="r")
N, J, D = acts.shape
L = J - 1  # number of blocks


def sel(ds, split=None, cond=None):
    return np.array([k for k, e in enumerate(ex) if e["dataset"] == ds and (split is None or e["split"] == split)
                     and (cond is None or cond(e))])


lab = np.array([e["label"] for e in ex])
items = np.array([e["item"] for e in ex])
topics = np.array([e["topic"] for e in ex])
para = np.array([e["para"] for e in ex])

I = dict(d1_tr=sel("d1", "train"), d1_te=sel("d1", "test"), d2_tr=sel("d2", "train"), d2_te=sel("d2", "test"),
         neg=sel("d1neg", "test"), nosys_tr=sel("d1nosys", "train"))
EVAL_SETS = {"d1_test": I["d1_te"], "d2_test": I["d2_te"], "d2_train": I["d2_tr"], "d1neg": I["neg"]}
for src in ("d1", "d2"):
    for fmt in ("mcq", "json", "plain"):
        EVAL_SETS[f"d3_{src}_{fmt}"] = sel(f"d3_{src}_{fmt}", "test")
fmt_tr = {f: sel(f"d3_d1_{f}", "train") for f in ("mcq", "json", "plain")}

# Hewitt & Liang-style control task (PREREG amendment 1). A "type" = (paraphrase string, first word of the
# task); both are visible in the input and occur in train AND test. Each type gets a fixed random label.
# Seeds are tried in order until the control labels agree with the true labels on 45-55% of D1 train,
# so the control task cannot borrow real signal.
verb = np.array([e["user"].split()[0].lower() for e in ex])
types = sorted({(int(p), str(v)) for p, v, e in zip(para, verb, ex) if e["dataset"] == "d1"})
for ctrl_seed in range(123, 1123):
    rng = np.random.default_rng(ctrl_seed)
    ctrl_of_type = {t: int(rng.integers(0, 2)) for t in types}
    ctrl = np.array([ctrl_of_type.get((int(p), str(v)), -1) for p, v in zip(para, verb)])
    agree_tr = (ctrl[I["d1_tr"]] == lab[I["d1_tr"]]).mean()
    bal_tr = ctrl[I["d1_tr"]].mean()
    if 0.45 <= agree_tr <= 0.55 and 0.4 <= bal_tr <= 0.6:
        break
ctrl_agree = float((ctrl[I["d1_te"]] == lab[I["d1_te"]]).mean())
ctrl_of_para = {f"{p}|{v}": l for (p, v), l in ctrl_of_type.items()}

# leave-paraphrase-out: train on paraphrase slots {0,1,2}, test on slots {3,4}
lpo_tr = I["d1_tr"][np.isin(para[I["d1_tr"]] % 5, [0, 1, 2])]
lpo_te = I["d1_te"][np.isin(para[I["d1_te"]] % 5, [3, 4])]

layers = []
dirs = {k: np.zeros((J, D), np.float32) for k in ("d1", "d2", "d1_deconf", "ctrl")}
deconf_basis = {}
K_LIST = [0, 1, 2, 4, 8, 16]

for j in range(J):
    X = np.asarray(acts[:, j, :], dtype=np.float32)
    R = dict(layer=j)
    Xtr, ytr = X[I["d1_tr"]], lab[I["d1_tr"]]
    # --- D1 mean-diff direction and logistic probe
    v1 = P.mean_diff(Xtr, ytr); dirs["d1"][j] = v1
    dm = P.Direction(v1, Xtr, ytr)
    pr = P.Probe(Xtr, ytr, groups=topics[I["d1_tr"]])
    R["probe_C"] = pr.C
    for name, idx in EVAL_SETS.items():
        R[f"md:{name}"] = P.summarize(dm.score(X[idx]), lab[idx], items[idx], thr=dm.thr)
        R[f"lr:{name}"] = P.summarize(pr.score(X[idx]), lab[idx], items[idx], pred=pr.predict(X[idx]))
    # --- D2-trained direction / probe (reference for transfer) and D2 -> D1
    X2, y2 = X[I["d2_tr"]], lab[I["d2_tr"]]
    v2 = P.mean_diff(X2, y2); dirs["d2"][j] = v2
    dm2 = P.Direction(v2, X2, y2)
    pr2 = P.Probe(X2, y2, groups=topics[I["d2_tr"]])
    for name in ("d2_test", "d1_test", "d1neg"):
        idx = EVAL_SETS[name]
        R[f"md2:{name}"] = P.summarize(dm2.score(X[idx]), lab[idx], items[idx], thr=dm2.thr)
        R[f"lr2:{name}"] = P.summarize(pr2.score(X[idx]), lab[idx], items[idx], pred=pr2.predict(X[idx]))
    # --- (d) control task, same probe hyper-parameters as the real probe
    yc = ctrl[I["d1_tr"]]
    prc = P.Probe(Xtr, yc, C=pr.C)
    vc = P.mean_diff(Xtr, yc); dirs["ctrl"][j] = vc
    dmc = P.Direction(vc, Xtr, yc)
    te = I["d1_te"]
    R["lr:ctrl_d1_test"] = P.summarize(prc.score(X[te]), ctrl[te], items[te], pred=prc.predict(X[te]))
    R["md:ctrl_d1_test"] = P.summarize(dmc.score(X[te]), ctrl[te], items[te], thr=dmc.thr)
    R["lr:ctrl_train_acc"] = float((prc.predict(Xtr) == yc).mean())
    # --- (e) confounds: paraphrase identity (10-way) and leave-paraphrase-out
    pid = P.Probe(Xtr, para[I["d1_tr"]], C=1.0)
    R["para_id_acc"] = P.acc_ci(pid.predict(X[te]) == para[te], items[te])
    lpo = P.Direction(P.mean_diff(X[lpo_tr], lab[lpo_tr]), X[lpo_tr], lab[lpo_tr])
    R["md:lpo_d1_test"] = P.summarize(lpo.score(X[lpo_te]), lab[lpo_te], items[lpo_te], thr=lpo.thr)
    lpo_lr = P.Probe(X[lpo_tr], lab[lpo_tr], C=pr.C)
    R["lr:lpo_d1_test"] = P.summarize(lpo_lr.score(X[lpo_te]), lab[lpo_te], items[lpo_te],
                                      pred=lpo_lr.predict(X[lpo_te]))
    # --- (f) cosine D1 vs D2, with a permutation null (shuffle D1 labels within each task pair)
    R["cos_d1_d2"] = P.cos(v1, v2)
    if a.n_perm:
        tr_items = items[I["d1_tr"]]
        uniq, inv = np.unique(tr_items, return_inverse=True)
        prng = np.random.default_rng(j)
        # NB: flipping labels *within* task pairs is not a valid null here: the result is a random +-1 sum of pair
        # differences that all share the true direction, so it stays partly aligned with +-v1. Use a global shuffle.
        null, null_au, rand_au = [], [], []
        X2te, y2te = X[I["d2_te"]], lab[I["d2_te"]]
        for _ in range(a.n_perm):
            yp = prng.permutation(ytr)
            vp = P.mean_diff(Xtr, yp)
            null.append(P.cos(vp, v2))
            null_au.append(P.auroc(X2te @ vp, y2te))
            rand_au.append(P.auroc(X2te @ prng.standard_normal(D), y2te))
        R["cos_null_abs95"] = float(np.percentile(np.abs(null), 95))

        def band(xs):
            dev = float(np.percentile(np.abs(np.array(xs) - 0.5), 95))
            return [0.5 - dev, 0.5 + dev]
        R["transfer_null_shuffle95"] = band(null_au)
        R["transfer_null_random95"] = band(rand_au)
        # one-sided empirical p: how often does a null direction rank D2 emotional-above-neutral at least as well?
        obs = P.auroc(X2te @ v1, y2te)
        R["transfer_p_shuffle"] = float((1 + np.sum(np.array(null_au) >= obs)) / (1 + len(null_au)))
        R["transfer_p_random"] = float((1 + np.sum(np.array(rand_au) >= obs)) / (1 + len(rand_au)))
    # --- deconfounding: format + paraphrase + instruction-presence subspace from D1 TRAIN only
    base = X[I["d1_tr"]].mean(0)
    vecs = [X[fmt_tr[f]].mean(0) - base for f in fmt_tr]
    for c in (0, 1):
        cls = I["d1_tr"][ytr == c]
        cm = X[cls].mean(0)
        for p in sorted(set(para[cls])):
            vecs.append(X[cls[para[cls] == p]].mean(0) - cm)
    vecs.append(base - X[I["nosys_tr"]].mean(0))
    # irrelevant-contrast directions (exploratory specificity check): how well do non-emotion contrasts built from
    # D1 data "transfer" to D2? Topic-vs-rest (10), format (3), paraphrase (10), instruction presence (1).
    irr_names = [f"format:{f}" for f in fmt_tr] + [f"para:{c}:{p}" for c in (0, 1)
                                                    for p in sorted(set(para[I["d1_tr"]][ytr == c]))] + ["presence"]
    irr = list(vecs)
    for tpc in sorted(set(topics[I["d1_tr"]])):
        m = topics[I["d1_tr"]] == tpc
        irr.append(Xtr[m].mean(0) - Xtr[~m].mean(0)); irr_names.append(f"topic:{tpc}")
    irr_au = [P.auroc(X[I["d2_te"]] @ w, lab[I["d2_te"]]) for w in irr]
    R["irrelevant_dirs_d2_auroc"] = dict(zip(irr_names, irr_au))
    R["irrelevant_dirs_max_dev"] = float(np.max(np.abs(np.array(irr_au) - 0.5)))
    _obs = abs(P.auroc(X[I["d2_te"]] @ v1, lab[I["d2_te"]]) - 0.5)
    R["irrelevant_dirs_frac_as_strong"] = float(np.mean(np.abs(np.array(irr_au) - 0.5) >= _obs))
    Bas, sv = P.orth_basis(np.stack(vecs))
    deconf_basis[j] = Bas
    R["subspace_rank"] = int(len(Bas))
    R["d1_frac_in_subspace"] = float(np.sum((Bas @ P.unit(v1)) ** 2))
    for k in K_LIST:
        kk = min(k, len(Bas))
        vk = P.project_out(v1, Bas[:kk]) if kk else v1
        dk = P.Direction(vk, Xtr, ytr)
        R[f"deconf_k{k}:d2_test"] = P.summarize(dk.score(X[I["d2_te"]]), lab[I["d2_te"]], items[I["d2_te"]], thr=dk.thr)
        R[f"deconf_k{k}:d1_test"] = P.summarize(dk.score(X[te]), lab[te], items[te], thr=dk.thr)
    vfull = P.project_out(v1, Bas); dirs["d1_deconf"][j] = vfull
    dfull = P.Direction(vfull, Xtr, ytr)
    for name, idx in EVAL_SETS.items():
        R[f"mdx:{name}"] = P.summarize(dfull.score(X[idx]), lab[idx], items[idx], thr=dfull.thr)
    # typical residual norm at the final prompt token (D1 train), for reference
    R["final_tok_norm_median"] = float(np.median(np.linalg.norm(Xtr, axis=1)))
    layers.append(R)
    print(f"layer {j:2d}  D1 acc md={R['md:d1_test']['acc'][0]:.2f} lr={R['lr:d1_test']['acc'][0]:.2f}  "
          f"D1->D2 auroc md={R['md:d2_test']['auroc'][0]:.2f} lr={R['lr:d2_test']['auroc'][0]:.2f}  "
          f"deconf={R['mdx:d2_test']['auroc'][0]:.2f}  D2 self={R['md2:d2_test']['auroc'][0]:.2f}  "
          f"ctrl={R['lr:ctrl_d1_test']['acc'][0]:.2f}  paraID={R['para_id_acc'][0]:.2f}  "
          f"cos={R['cos_d1_d2']:+.2f}  ({time.time()-t0:.0f}s)", flush=True)

# full layer x layer cosine matrix D1 vs D2
cosmat = np.array([[P.cos(dirs["d1"][i], dirs["d2"][k]) for k in range(J)] for i in range(J)])

# --- lexical (text-only) confound baseline: TF-IDF logistic on the raw prompt text
def text_of(e):
    return ((e["system"] or "") + " \n " + e["user"]).strip()

tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
Ttr = tf.fit_transform([text_of(ex[k]) for k in I["d1_tr"]])
lex = LogisticRegression(C=1.0, max_iter=3000).fit(Ttr, lab[I["d1_tr"]])
lexical = {}
for name, idx in EVAL_SETS.items():
    T = tf.transform([text_of(ex[k]) for k in idx])
    lexical[name] = P.summarize(lex.decision_function(T), lab[idx], items[idx], pred=lex.predict(T))
# lexical D2 reference: text classifier trained on D2 train
tf2 = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
lex2 = LogisticRegression(C=1.0, max_iter=3000).fit(tf2.fit_transform([text_of(ex[k]) for k in I["d2_tr"]]), lab[I["d2_tr"]])
T2 = tf2.transform([text_of(ex[k]) for k in I["d2_te"]])
lexical["d2_trained_on_d2_test"] = P.summarize(lex2.decision_function(T2), lab[I["d2_te"]], items[I["d2_te"]],
                                               pred=lex2.predict(T2))

# --- pre-registered steering layer: argmax D1 mean-diff transfer AUROC on D2 TRAIN within [L/4, 2L/3]
lo, hi = int(np.ceil(L / 4)), int(np.floor(2 * L / 3))
cand = list(range(max(lo, 1), hi + 1))
au = [layers[j]["md:d2_train"]["auroc"][0] for j in cand]
steer_layer = cand[int(np.argmax(au))]  # argmax returns the first (lowest) on ties

# per-emotion transfer at the steering layer (exploratory)
X = np.asarray(acts[:, steer_layer, :], dtype=np.float32)
dm = P.Direction(dirs["d1"][steer_layer], X[I["d1_tr"]], lab[I["d1_tr"]])
per_emotion = {}
te2 = I["d2_te"]
for emo in ("grief", "joy", "fear", "anger"):
    its = {items[k] for k in te2 if ex[k]["emotion"] == emo}
    idx = np.array([k for k in te2 if items[k] in its])
    per_emotion[emo] = P.summarize(dm.score(X[idx]), lab[idx], items[idx], thr=dm.thr)
    per_emotion[emo]["n_pairs"] = len(its)

# length-only baseline (exploratory): does prompt length alone separate D2 emotional vs neutral?
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained(a.model)
length_baseline = {}
for name in ("d2_test", "d2_train", "d1_test"):
    idx = EVAL_SETS[name]
    nt = np.array([len(_tok(((ex[k]["system"] or "") + " " + ex[k]["user"]).strip(), add_special_tokens=False)["input_ids"])
                   for k in idx], float)
    length_baseline[name] = P.summarize(nt, lab[idx], items[idx])
# length-residualised D1->D2 transfer at every layer (exploratory): regress the D1-direction score on token count
# (pooled D2 test), then AUROC of the residual.
nt2 = np.array([len(_tok(ex[k]["user"], add_special_tokens=False)["input_ids"]) for k in I["d2_te"]], float)
A_ = np.stack([nt2, np.ones_like(nt2)], 1)
for j in range(J):
    s = np.asarray(acts[I["d2_te"], j, :], dtype=np.float32) @ P.unit(dirs["d1"][j])
    resid = s - A_ @ np.linalg.lstsq(A_, s, rcond=None)[0]
    layers[j]["md:d2_test_len_resid"] = P.summarize(resid, lab[I["d2_te"]], items[I["d2_te"]])

out = dict(model=a.model, n_blocks=L, d_model=D, control_label_agreement_with_true=ctrl_agree,
           control_seed=ctrl_seed, control_of_type=ctrl_of_para, length_baseline_auroc=length_baseline, steer_layer=steer_layer, steer_layer_candidates=cand,
           steer_layer_candidate_auroc_d2_train=au, lexical_baseline=lexical, per_emotion_at_steer_layer=per_emotion,
           cos_matrix_d1_rows_d2_cols=cosmat.tolist(), layers=layers, k_list=K_LIST)
(RESULTS / S).mkdir(parents=True, exist_ok=True)
(RESULTS / S / "probe_metrics.json").write_text(json.dumps(out, indent=1))
np.savez(CACHE / S / "directions.npz", **dirs, steer_layer=steer_layer,
         deconf_basis_steer=deconf_basis[steer_layer])
print(f"steer layer = {steer_layer} (candidates {cand}); lexical D1 test acc {lexical['d1_test']['acc'][0]:.2f}, "
      f"lexical D1->D2 auroc {lexical['d2_test']['auroc'][0]:.2f}; total {time.time()-t0:.0f}s")
