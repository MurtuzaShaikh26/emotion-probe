"""Figures F1-F5 (PNG + PDF, 300 dpi) from results/<slug>/*.json. Usage: python scripts/make_figures.py llama8b qwen3b"""
import json, sys, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from emoprobe.datasets import load_jsonl
from emoprobe.paths import RESULTS, FIGS

# validated categorical slots (dataviz reference palette, light mode) + neutral gray for controls
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781", "#0b0b0b", "#52514e"
VEC_STYLE = {"d1": (BLUE, "o", "-", "D1 direction (instructions)"),
             "d2": (ORANGE, "s", "-", "D2 direction (content)"),
             "d1_deconf": (AQUA, "^", "-", "D1 minus format subspace"),
             "random": (GRAY, "D", "--", "Random direction (control)")}
NAMES = {"llama8b": "Llama-3.1-8B-Instruct", "qwen3b": "Qwen2.5-3B-Instruct"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "legend.frameon": False, "lines.linewidth": 2})
FIGS.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def series(layers, key, metric):
    arr = np.array([L[key][metric] for L in layers])
    return arr[:, 0], arr[:, 1], arr[:, 2]


def band_line(ax, x, m, lo, hi, color, label, ls="-", marker=None):
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=color, ls=ls, label=label, marker=marker, markersize=3 if marker else 0)


def f1(S, pm):
    Ls = pm["layers"]; x = np.arange(len(Ls)); js = pm["steer_layer"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    band_line(a1, x, *series(Ls, "md:d1_test", "acc"), BLUE, "D1 held-out, mean-diff direction")
    band_line(a1, x, *series(Ls, "lr:d1_test", "acc"), BLUE, "D1 held-out, logistic probe", ls=":")
    band_line(a1, x, *series(Ls, "lr:ctrl_d1_test", "acc"), GRAY, "Control task (random labels per type), probe", ls="--")
    pid = np.array([L["para_id_acc"] for L in Ls])
    band_line(a1, x, pid[:, 0], pid[:, 1], pid[:, 2], YELLOW, "Which paraphrase? (10-way, chance 0.1)", ls="-.")
    a1.set(xlabel="layer (residual index)", ylabel="accuracy", ylim=(0, 1.03), title="(a) Reading the instruction contrast (D1)")
    a1.legend(fontsize=8, loc="lower right")
    lo_r = [L["transfer_null_random95"][0] for L in Ls]; hi_r = [L["transfer_null_random95"][1] for L in Ls]
    a2.fill_between(x, lo_r, hi_r, color=GRAY, alpha=0.18, linewidth=0, label="95% band: random directions")
    band_line(a2, x, *series(Ls, "md:d2_test", "auroc"), BLUE, "D1 direction → D2 (emotional content)")
    d3i = np.mean([series(Ls, f"md:d3_d1_{f}", "auroc")[0] for f in ("mcq", "json", "plain")], 0)
    d3c = np.mean([series(Ls, f"md:d3_d2_{f}", "auroc")[0] for f in ("mcq", "json", "plain")], 0)
    a2.plot(x, d3i, color=AQUA, ls=":", label="D1 direction → D3 instruction contrast, new formats (mean of 3)")
    a2.plot(x, d3c, color=AQUA, ls="-", label="D1 direction → D3 content contrast, new formats (mean of 3)")
    a2.plot(x, series(Ls, "md2:d2_test", "auroc")[0], color=ORANGE, ls="--", label="Reference: D2 direction → D2")
    a2.axhline(0.5, color=INK2, lw=0.8); a2.axvline(js, color=INK2, lw=0.8, ls=":")
    a2.annotate(f"steering layer {js}", (js, 0.03), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2)
    a2.set(xlabel="layer (residual index)", ylabel="AUROC (threshold-free)", ylim=(0, 1.03), title="(b) Transfer of the D1 direction")
    a2.legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"F1 · Probe accuracy and transfer by layer · {NAMES[S]}", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, f"F1_probe_by_layer_{S}")


def f2(S, pm):
    Ls = pm["layers"]; x = np.arange(len(Ls))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    cs = np.array([L["cos_d1_d2"] for L in Ls]); nb = np.array([L.get("cos_null_abs95", np.nan) for L in Ls])
    a1.fill_between(x, -nb, nb, color=GRAY, alpha=0.2, linewidth=0, label="95% band: D1 with shuffled labels")
    a1.plot(x, cs, color=BLUE, marker="o", markersize=3, label="cos(D1 direction, D2 direction), same layer")
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set(xlabel="layer (residual index)", ylabel="cosine similarity", ylim=(-1, 1), title="(a) Same-layer cosine")
    a1.legend(fontsize=8, loc="upper left")
    M = np.array(pm["cos_matrix_d1_rows_d2_cols"])
    im = a2.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    a2.set(xlabel="D2 direction, layer", ylabel="D1 direction, layer", title="(b) All layer pairs")
    a2.grid(False); fig.colorbar(im, ax=a2, fraction=0.046, label="cosine")
    fig.suptitle(f"F2 · Do instructed and content-based emotion directions align? · {NAMES[S]}", x=0.01, ha="left",
                 fontsize=12, color=INK)
    save(fig, f"F2_cosine_d1_d2_{S}")


def dose_plot(ax, sm, key, cs, transform=lambda v: v):
    for v, (col, mk, ls, lab) in VEC_STYLE.items():
        arr = np.array([sm["dose"][key][v][str(c)] for c in cs], dtype=float)
        m, lo, hi = transform(arr[:, 0]), transform(arr[:, 1]), transform(arr[:, 2])
        off = {"d1": -0.12, "d2": 0.12, "d1_deconf": 0.0, "random": 0.24}[v]
        ax.errorbar(np.array(cs) + off, m, yerr=[m - lo, hi - m], color=col, marker=mk, ls=ls, markersize=6,
                    capsize=2, label=lab, lw=2)
    ax.set_xticks(cs); ax.axvline(0, color=INK2, lw=0.6)


def f3(S, sm):
    cs = [-8, -4, -2, 0, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    dose_plot(ax, sm, "emotion", cs)
    ax.set(xlabel="steering strength c   (α = c × %.2f, added at every position, layer %d)" % (sm["alpha_per_c"], sm["layer"]),
           ylabel="judge emotional intensity (1–10)", ylim=(0.8, 10.2),
           title=f"F3 · Steering dose-response · {NAMES[S]}  (mean ± 95% CI, n={sm['n_prompts']} prompts)")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, f"F3_steering_dose_response_{S}")


def f4(S, sm):
    cs = [-8, -4, -2, 0, 2, 4, 8]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    dose_plot(a1, sm, "log_ppl", cs, transform=np.exp)
    a1.set_yscale("log"); a1.set(xlabel="steering strength c", ylabel="perplexity under unsteered model (geo. mean)",
                                 title="(a) Fluency cost")
    dose_plot(a2, sm, "task_completed", cs)
    a2.set(xlabel="steering strength c", ylabel="judge: task completed (rate)", ylim=(0, 1.05), title="(b) Task completion")
    a2.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"F4 · What steering costs · {NAMES[S]}", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, f"F4_fluency_task_cost_{S}")


def f5(S, c=4, vector="d1"):
    gens = load_jsonl(RESULTS / S / "generations.jsonl")
    pids = sorted({g["prompt_id"] for g in gens}, key=lambda p: (p.split(":")[1], int(p.split(":")[2])))
    order = [g for g in load_jsonl(Path(__file__).resolve().parents[1] / "data/examples.jsonl") if g["dataset"] == "steer"]
    first3 = [e["id"] for e in order[:3]]  # fixed rule: first 3 steering prompts in dataset order
    get = {(g["prompt_id"], g["vector"], g["c"]): g for g in gens}
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5))
    for r, pid in enumerate(first3):
        for k, cc in enumerate((-c, 0, c)):
            g = get[(pid, "none", 0)] if cc == 0 else get[(pid, vector, cc)]
            ax = axes[r, k]; ax.axis("off")
            txt = g["text"].replace("\n", " ")
            txt = txt if len(txt) <= 420 else txt[:420] + " […]"
            ax.text(0, 1, "\n".join(textwrap.wrap(txt, 58)), va="top", ha="left", fontsize=8.2, color=INK,
                    family="DejaVu Sans", transform=ax.transAxes)
            if r == 0:
                ax.set_title(f"c = {cc:+d}" + ("  (unsteered)" if cc == 0 else ""), fontsize=11, color=INK, loc="left")
        axes[r, 0].annotate(f"Prompt: {g['request']}", (0, 1.07), xycoords="axes fraction", fontsize=9.5,
                            color=INK2, fontweight="bold")
    fig.suptitle(f"F5 · First 3 held-out prompts (fixed rule), steered with the {VEC_STYLE[vector][3]} at c = ±{c} · "
                 f"{NAMES[S]} · greedy, 128 tokens, text truncated for display", x=0.01, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(hspace=0.35, wspace=0.08)
    save(fig, f"F5_qualitative_{vector}_c{c}_{S}")


for S in sys.argv[1:] or ["llama8b", "qwen3b"]:
    pmf, smf = RESULTS / S / "probe_metrics.json", RESULTS / S / "steering_metrics.json"
    if pmf.exists():
        pm = json.loads(pmf.read_text()); f1(S, pm); f2(S, pm)
    if smf.exists():
        sm = json.loads(smf.read_text())
        if sm["n_judged"]:
            f3(S, sm)
        f4(S, sm) if sm["n_judged"] else None
    if (RESULTS / S / "generations.jsonl").exists():
        f5(S, 4, "d1"); f5(S, 8, "d1"); f5(S, 4, "d2")
    print("figures done for", S)
