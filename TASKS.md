# Task list

Legend: [x] done · [ ] to do · [~] in progress · (M) = needs Murt

## Stage 0 — Setup
- [x] Write CLAUDE.md with full brief
- [x] Write this task list
- [x] git init + remote; .gitignore; .env template
- [x] Modal CLI installed and token verified (profile ronitk964)
- [ ] (M) FIREWORKS_API_KEY in .env
- [ ] First push to GitHub

## Stage 0.5 — Data
- [ ] D1: ~150 neutral tasks in topic groups; 5 empathetic + 5 cold system-prompt paraphrases, rotated
- [ ] D1-neg: negated-instruction control (tone words present, meaning flipped) — confound test
- [ ] D1-nosys: tasks with no system prompt (for the instruction-presence direction)
- [ ] D2: ~150 topic-matched pairs (grief/joy/fear/anger vs neutral), length-matched, no tone system prompt
- [ ] D3: MCQ / JSON / plain-completion variants of D1 and D2
- [ ] 60 held-out neutral steering prompts + calibration prompts
- [ ] Topic-based train/test split; leakage-check script

## Stage 0.75 — Local prototype (Qwen2.5-3B) + unit tests
- [ ] Activation extraction with HF hooks (all layers, final prompt token), cached to disk
- [ ] Unit tests: hook shapes, token position, label alignment, leakage, c=0 reproduces baseline
- [ ] Probe pipeline end-to-end on Qwen (prototype numbers, not reported as main results)
- [ ] Steering + perplexity pipeline end-to-end on Qwen (small smoke run)

## Stage 0.9 — Pre-registration
- [ ] docs/PREREG.md (H1–H4, metrics, thresholds, layer-choice rule, scale-calibration rule) — committed before main runs

## Stage 1 — Reading (Llama-3.1-8B on Modal; Qwen replication locally)
- [ ] Extract activations (one Modal call)
- [ ] a. D1 held-out accuracy (mean-diff + logistic), bootstrap CIs
- [ ] b. D1→D2 transfer
- [ ] c. D1→D3 transfer
- [ ] d. Hewitt & Liang control task + selectivity
- [ ] e. Confound probes (paraphrase-ID, leave-paraphrase-out, lexical text baseline, negation test)
- [ ] f. D1 vs D2 cosine matrix (+ permutation null)
- [ ] Deconfounding (format/instruction subspace projection), transfer vs k

## Stage 2 — Control (steering)
- [ ] Choose layer by pre-registered rule; calibrate scale
- [ ] Generate: 60 prompts × {D1, D2, D1-deconf, random} × c ∈ {±2, ±4, ±8} + c=0 baseline
- [ ] Perplexity under unsteered model; regex side effects
- [ ] Judge (Fireworks gpt-oss-120b, T=0, blind, shuffled)
- [ ] (M) Hand-label 40 outputs → judge–human agreement
- [ ] Qwen replication of steering

## Stage 3 — Outputs
- [ ] F1–F5 (PNG + PDF, 300 dpi)
- [ ] results/RESULTS.md (generated) + JSON metrics + JSONL generations
- [ ] docs/SUMMARY.md (≤15 lines)
- [ ] README.md (plain language, limitations)
