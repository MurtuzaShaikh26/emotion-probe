# Pre-registration

Written 2026-11-03, **before** any probe or steering numbers were inspected (for either model).
Committed to git before the main (Llama) runs; the commit hash is the timestamp. Anything not listed
here is **exploratory** and labelled so in all outputs.

## Question
Does an LLM have a linear "emotional tone" direction that causally controls its outputs, or does a
contrastive probe built from instructed prompts mostly detect the instruction text?

## Models
- Main: Llama-3.1-8B-Instruct (`unsloth/Meta-Llama-3.1-8B-Instruct`, a byte-identical ungated mirror), fp16.
- Replication: Qwen2.5-3B-Instruct, fp16. Same code, same rules; layer ranges scale with depth.

## Data (built by `scripts/build_datasets.py`, fixed seed 0)
- D1: 150 neutral tasks (15 topic groups) × {empathetic, cold} system prompt; 5 paraphrases per class rotated
  by task. Train = 10 topic groups (200 prompts), test = 5 groups (100 prompts).
- D2: 150 topic-matched pairs (30 topic groups; grief/joy/fear/anger vs neutral), no tone system prompt.
  Train = 21 groups (210 prompts), test = 9 groups (90 prompts).
- D3: D1 and D2 test items in MCQ, JSON-request, and plain-completion (no chat template) formats.
- D1-neg (confound control): D1 test tasks with negated instructions ("Do not be warm, caring…" = cold,
  label 0; "Do not be cold, robotic…" = warm, label 1).
- Steering: 60 new neutral prompts from 6 topic groups not used anywhere else. Calibration: 10 D1-train tasks.

## Representations
Residual stream at the final prompt token (last token of the chat template's assistant header), every
layer j = 0..L (j = 0 embeddings, j = i+1 output of block i).

## Readers
- Direction: mean(emotional) − mean(neutral) on D1 train. Score = projection; accuracy threshold = midpoint of
  the two projected training class means.
- Probe: standardise, L2 logistic regression; C ∈ {0.001, 0.01, 0.1, 1} by 5-fold group-CV over topics on train.
- **Primary transfer metric is AUROC** (threshold-free), because a shift in mean activation between datasets
  (e.g. no system prompt in D2) can push every example to one side of a D1-fitted threshold even when the
  direction still ranks emotional above neutral. Accuracy with the D1 threshold is also reported.
- 95% CIs: cluster bootstrap over items (a task's two prompts, or a D2 pair, resampled together), 1000 draws.

## Hypotheses and decision rules
- **H1 (D1 decodable mid-to-late).** Supported if, for at least half of layers j ∈ [L/3, L], both the
  direction and the probe have held-out D1 accuracy with 95% CI lower bound > 0.80.
- **H2 (D1 → D2 transfer above chance).** Evaluated at the steering layer j* (chosen below on D2 *train*),
  on D2 *test*. Supported if the D1 direction's AUROC 95% CI lower bound > 0.50. Reported qualitatively as
  "weak" (point < 0.65), "moderate" (0.65–0.80) or "strong" (≥ 0.80).
- **H3 (D1 steering raises emotion vs random, task survives).** At c = +4: the mean paired difference
  (D1 − random) in judge emotional-intensity over the 60 prompts has 95% bootstrap CI lower bound > 0, AND the
  judge task-completion rate for D1 at c = +4 is ≥ 0.70.
- **H4 (D2 steers at least as well as D1).** At c = +4: the lower 95% CI bound of the paired difference
  (D2 − D1) in emotional intensity is > −1.0 (non-inferiority margin of 1 point), AND D2 − random has CI
  lower bound > 0.

## Steering protocol
- **Layer rule:** j* = argmax over j ∈ [⌈L/4⌉, ⌊2L/3⌋] of the D1 direction's AUROC on **D2 train**
  (lowest layer wins ties). Chosen by transfer, not by D1 accuracy (D1 accuracy is expected to saturate and so
  cannot discriminate layers). Using D2 train for selection keeps D2 test clean for H2.
- **Vectors** (all unit-normalised at j*): (i) D1 direction; (ii) D2 direction (D2 train); (iii) D1 direction
  with the format/instruction subspace projected out (full span — see below); (iv) a Gaussian random unit
  vector (seed 0). The same α is used for all four.
- **Scale calibration (fluency only, never judge scores):** typical norm ν = median per-token residual norm at
  j* over the 10 calibration prompts (first token excluded). For f ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8,
  1.0}, steer the calibration prompts with ±f·ν·(D1 direction), 64 greedy tokens. Breaking point f* = smallest
  f where, for either sign, mean perplexity (under the unsteered model) > 3× the unsteered mean, or mean
  distinct-bigram ratio < 0.5. If none breaks, f* = 1.0. Then α(c) = c · f*·ν / 8, so c = ±8 sits at the
  breaking point and c = ±2 at a quarter of it.
- **Grid:** c ∈ {−8, −4, −2, 0, 2, 4, 8}; added at all positions (prompt and generated); greedy decoding;
  128 new tokens; chat template with no system prompt. c = 0 is generated once and shared by all vectors.

## Format/instruction subspace (deconfounding)
At each layer, from D1 train only: 3 format directions (mean of MCQ/JSON/plain variants − mean of chat D1),
10 paraphrase directions (paraphrase mean − its class mean), 1 instruction-presence direction (mean of D1 −
mean of the same tasks with no system prompt). Unit-normalise, SVD, keep all non-degenerate components.
v_deconf = v_D1 minus its projection onto that span. D1→D2 transfer is also reported for the top k ∈ {0, 1, 2, 4,
8, 16} components (exploratory).

## Judge
Fireworks `gpt-oss-120b`, temperature 0, one call per output, outputs shuffled with seed 0, judge never sees
the condition, vector or c. Frozen rubric (text in `src/emoprobe/judge.py`, `RUBRIC`):
- emotional_intensity 1–10 (1 = flat/factual, 10 = overwhelmingly emotional; valence ignored);
- task_completed yes/no (genuine relevant attempt at the request; truncation at the length limit is not penalised);
- refusal yes/no.
Judge validation: Murt hand-labels 40 outputs (stratified random sample, seed 0, blind). Report Spearman ρ and
mean absolute difference for intensity, and percent agreement + Cohen's κ for task completion.

## Other measures (all descriptive)
Perplexity of each output under the unsteered model; regex rates of refusal phrases, apology words and
bullet/numbered lists; Hewitt & Liang control task (labels assigned at random to each of the 10 paraphrase
strings) and selectivity = real − control accuracy; paraphrase-identity probe; leave-paraphrase-out accuracy;
TF-IDF text-only baseline; negation test; D1–D2 cosine with a label-permutation null.

## Amendments (made before any Llama run)
1. **Control task types.** On the Qwen prototype, giving each of the 10 paraphrase strings a random label
   produced control labels that agreed with the true labels on 80% of examples, so the control task borrowed real
   signal. With 5 paraphrases per class, per-paraphrase labels can never be independent of the class. New rule:
   a type = (paraphrase string, first word of the task, e.g. "Explain"); each type gets a fixed random label;
   seeds are tried from 123 upward until agreement with the true labels on D1 train is within [0.45, 0.55] and
   the label balance is within [0.4, 0.6]. Both type parts are visible in the input and occur in both splits, as in
   Hewitt & Liang. This changes no hypothesis or threshold.
2. **Length baseline** (exploratory addition): AUROC of prompt token count alone, plus D1→D2 transfer after
   regressing token count out of the projection score.
3. **D2 length re-matching (data change).** The Qwen prototype showed word count alone separated D2 emotional vs
   neutral messages with AUROC 0.79, so "transfer" could have been a length detector. Before any Llama run, the
   neutral side of 140 pairs was rewritten (same topic, still neutral) so every pair is within ±1 **token**
   (Llama and Qwen tokenizers); token-length AUROC is now 0.52 (`scripts/check_lengths.py`). The emotional
   messages, topics, splits and all hypotheses are unchanged. Qwen numbers from before this fix are discarded.
4. **Transfer specificity test (added after seeing Qwen, before any Llama run).** On Qwen, D2 classes are so far
   apart at the final token that *random* directions often separate them (95% band of random-direction AUROC
   ≈ [0.08, 0.92] at late layers). So an AUROC CI above 0.5 is not, by itself, evidence that the D1 direction
   carries emotion. H2 is kept exactly as stated, and a secondary pre-specified test is added for Llama:
   **H2-specific** is supported if, at j*, the D1 direction's D2-test AUROC beats 1000 isotropic random directions
   one-sided at p < 0.05 (`transfer_p_random`). Also reported: p vs 1000 label-shuffled D1 directions, and the
   share of 24 irrelevant-contrast directions (topic, format, paraphrase, instruction presence) that separate
   D2 at least as strongly (sign ignored). The overall verdict rule reads "H2 moderate/strong" as "H2 moderate/strong
   AND H2-specific".

## Overall verdict rule (for the one-line summary)
- "emotion direction": H2 moderate/strong AND H3 supported.
- "mostly instruction detector": H2 not supported or weak, AND (H3 not supported OR D2 direction steers clearly
  better than D1, i.e. D2 − D1 CI lower bound > 0).
- otherwise "mixed".
