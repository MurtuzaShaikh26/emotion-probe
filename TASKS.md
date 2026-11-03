# Task list

Legend: [x] done · [ ] to do · [~] in progress · (M) = needs Murt

## Stage 0 — Setup
- [x] Write CLAUDE.md with full brief
- [x] Write this task list
- [x] git init + remote; .gitignore; .env template
- [x] Modal CLI installed and token verified (profile ronitk964)
- [ ] (M) FIREWORKS_API_KEY in .env
- [x] First push to GitHub

## Stage 0.5 — Data
- [x] D1: ~150 neutral tasks in topic groups; 5 empathetic + 5 cold system-prompt paraphrases, rotated
- [x] D1-neg: negated-instruction control (tone words present, meaning flipped) — confound test
- [x] D1-nosys: tasks with no system prompt (for the instruction-presence direction)
- [x] D2: ~150 topic-matched pairs (grief/joy/fear/anger vs neutral), length-matched, no tone system prompt
- [x] D3: MCQ / JSON / plain-completion variants of D1 and D2
- [x] 60 held-out neutral steering prompts + calibration prompts
- [x] Topic-based train/test split; leakage-check script

## Stage 0.75 — Local prototype (Qwen2.5-3B) + unit tests
- [x] Activation extraction with HF hooks (all layers, final prompt token), cached to disk
- [x] Unit tests: hook shapes, token position, label alignment, leakage, c=0 reproduces baseline
- [x] Probe pipeline end-to-end on Qwen (prototype numbers, not reported as main results)
- [x] Steering + perplexity pipeline end-to-end on Qwen (small smoke run)

## Stage 0.9 — Pre-registration
- [x] docs/PREREG.md (H1–H4, metrics, thresholds, layer-choice rule, scale-calibration rule) — committed before main runs

## Stage 1 — Reading (Llama-3.1-8B on Modal; Qwen replication locally)
- [x] Extract activations (one Modal call)
- [x] a. D1 held-out accuracy (mean-diff + logistic), bootstrap CIs
- [x] b. D1→D2 transfer
- [x] c. D1→D3 transfer
- [x] d. Hewitt & Liang control task + selectivity
- [x] e. Confound probes (paraphrase-ID, leave-paraphrase-out, lexical text baseline, negation test)
- [x] f. D1 vs D2 cosine matrix (+ permutation null)
- [x] Deconfounding (format/instruction subspace projection), transfer vs k

## Stage 2 — Control (steering)
- [x] Choose layer by pre-registered rule; calibrate scale
- [x] Generate: 60 prompts × {D1, D2, D1-deconf, random} × c ∈ {±2, ±4, ±8} + c=0 baseline
- [x] Perplexity under unsteered model; regex side effects
- [ ] (M) Judge (Fireworks — needs FIREWORKS_API_KEY; gpt-oss-120b, T=0, blind, shuffled)
- [ ] (M) Hand-label 40 outputs → judge–human agreement
- [x] Qwen replication of steering

## Stage 3 — Outputs
- [~] F1–F5 (PNG + PDF, 300 dpi) — F1, F2, F5 done; F3/F4 need judge scores
- [ ] results/RESULTS.md (generated) + JSON metrics + JSONL generations
- [ ] docs/SUMMARY.md (≤15 lines)
- [ ] README.md (plain language, limitations)

## Extra (found along the way)
- [x] D2 length confound found + fixed (PREREG amendment 3)
- [x] Transfer-specificity nulls: random / shuffled / irrelevant-contrast directions (amendment 4)
- [x] Negation test, lexical baseline, length-residualised transfer
