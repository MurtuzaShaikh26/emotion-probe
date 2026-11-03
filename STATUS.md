# STATUS

_Last updated: 2026-11-03_

## Stage 0 — setup ✅
Repo, CLAUDE.md, TASKS.md, .gitignore, `.env` template. Modal authenticated (profile `ronitk964`).
Models: Qwen2.5-3B-Instruct (local, RTX 5060) and `unsloth/Meta-Llama-3.1-8B-Instruct` (Modal A10G; an ungated
mirror, so **no Hugging Face token needed**).

## Stage 0.5 — data ✅
`data/examples.jsonl` (2,090 rows), built by `scripts/build_datasets.py`; stats in `results/dataset_stats.json`.
Topic-based splits; leakage + label checks pass (`tests/test_data.py`).
**Data fix found on the prototype:** token count alone separated D2 emotional vs neutral (AUROC 0.79 by words).
Neutral texts were rewritten so every pair is within ±1 token. Token-length AUROC is now 0.52
(`scripts/check_lengths.py`, PREREG amendment 3).

## Stage 0.75 — prototype + unit tests ✅
12/12 tests pass locally (Qwen) **and on Llama on Modal**: hook shapes, hooks == HF hidden_states, token position,
batch/padding invariance and row alignment, exact c=0 reproduction under greedy decoding, exact vector addition,
perplexity sanity.

## Stage 0.9 — pre-registration ✅
`docs/PREREG.md`, committed before any results were viewed (commit 0236ee0). Amendments 1–4 were committed
before any Llama run (a35761f): a control-task fix, a length baseline, D2 length re-matching, and a
transfer-specificity test vs random directions.

## Stage 1 — reading
- Qwen (replication): done. `results/qwen3b/probe_metrics.json`.
- Llama: activations extracted on Modal (88 s A10G ≈ $0.03); probe analysis running locally.

## Stage 2 — steering
- Qwen: 1,500 generations done locally (`results/qwen3b/generations.jsonl`). Judge blocked on FIREWORKS_API_KEY.
- Llama: waiting on Stage 1 → steering layer → Modal stage 2.

## Spend
Modal ≈ $0.03 so far (`results/costs.json`). Fireworks $0.
