# STATUS

_Last updated: 2026-11-03_

## Done
- Setup, data (2,090 rows, no leakage, D2 length confound fixed), 12/12 unit tests on Qwen (local) and Llama (Modal).
- Pre-registration `docs/PREREG.md` (amendments 1–4 committed before any Llama run).
- Stage 1 (reading) and Stage 2 (steering) on Llama-3.1-8B and Qwen2.5-3B.
- **Judge:** 3,000/3,000 generations scored by gpt-oss-120b on Fireworks (0 failures).
- **Outputs:** F1–F5 (PNG + PDF), `results/RESULTS.md`, `docs/SUMMARY.md`, `README.md`.
- **Second-rater check:** Claude Opus 5.5 labelled the 40 validation outputs blind (`results/claude_labels.csv`),
  reported as an LLM-vs-LLM check, **not** human validation.

## Headline (Llama, layer 18; details in docs/SUMMARY.md)
- H1 supported (uninformative: text alone gives 1.00). H2 supported: D1→D2 AUROC 0.96 [0.93, 0.99], beats random
  directions (p = 0.008), not shuffled-label directions (p = 0.16).
- H3 supported: D1 steering − random = +0.90 [0.65, 1.15] at c = +4, task completion 1.00.
- H4 not supported: the D2 (content) direction barely steers (D2 − random at +4 = +0.02 [−0.12, 0.17]).
- Pre-registered verdict: Llama "emotion direction", Qwen "mixed". Honest overall reading: **mixed**, i.e. an
  "act warm" style direction rather than a general emotion direction.

## Judge validation
Murt's 40 hand labels: judge Spearman 0.61, within ±1 on 93%, bias +0.10; judge compresses extremes.
Nothing open.

## Spend
Modal ≈ $0.15 of the $10 cap (`results/costs.json`). Fireworks ≈ $0.36 of the $8 cap (`results/judge_costs.json`).
