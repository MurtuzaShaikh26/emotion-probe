# STATUS

_Last updated: 2026-11-03_

## Done
- **Setup:** Modal authenticated (`ronitk964`). Main model is `unsloth/Meta-Llama-3.1-8B-Instruct`, an ungated mirror, so **no HF token is needed**.
- **Data:** 2,090 rows, topic-based splits, no leakage. The D2 length confound was found on the prototype and fixed (token-length AUROC now 0.52).
- **Tests:** 12/12 pass locally on Qwen **and on Llama on Modal**, including exact c = 0 reproduction.
- **Pre-registration:** `docs/PREREG.md` was committed before any results. Amendments 1–4 were committed before any Llama run.
- **Stage 1 (reading), Llama and Qwen:** done. See `results/RESULTS.md`, `figures/F1_*`, `figures/F2_*`.
- **Stage 2 generations:** Llama (Modal) and Qwen (local), 1,500 each, with perplexity and regex side effects. See `figures/F5_*`.

## Headline so far (Llama, steering layer j* = 18; full numbers in results/RESULTS.md)
- **H1 supported.** D1 is ~100% decodable mid-to-late, but so is it from the text alone (TF-IDF: 1.00). A probe can also name which paraphrase was used (0.98), and the control task reaches 0.78.
- **H2 supported; H2-specific supported.** D1→D2 AUROC is 0.96 [0.93, 0.99] and beats random directions (p = 0.008), but it does not beat label-shuffled D1 directions (p = 0.16). Some irrelevant contrasts (instruction present vs absent) separate D2 just as well.
- **Negation test:** the D1 direction follows instruction *meaning*, not tone words (1.00).
- **Deconfounding** barely changes transfer: 0.96 → 0.92.
- **Steering:** the D1 direction visibly makes text gushier at +4/+8; the D2 direction barely changes Llama's text at ±4.

## Blocked on Murt
1. FIREWORKS_API_KEY in `.env` → judge (H3/H4, F3, F4).
2. Hand labels in `results/human_labels.csv` → judge validation.

## Spend
Modal ≈ $0.15 of the $10 cap (`results/costs.json`). Fireworks: $0 so far.
