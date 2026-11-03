# CLAUDE.md — Emotion-probe project

This file holds the full project brief (verbatim) and the working decisions taken
while executing it. Task list: `TASKS.md`. Progress log: `STATUS.md`.

## Working rules (short form)
- Autonomous research engineer; explain choices so Murt learns (first hands-on interp project, working through ARENA).
- Ask Murt only for things only he can provide, in a "NEED FROM MURT" block with numbered steps.
- Secrets live in `.env` (gitignored) or `~/.modal.toml` — never committed.
- Long jobs run in the background with logs (`logs/`) and checkpoints; activations cached to `cache/` (gitignored).
- Update `STATUS.md` after each stage; commit after each stage; push to https://github.com/MurtuzaShaikh26/emotion-probe.
- Never fabricate numbers: every number in a doc must trace to a file in `results/`.
- Budget caps: Modal $10 (user asked to be frugal), Fireworks $8. Track spend in `results/costs.json`.
- No overclaiming ("proves", "sociopathic", "brain"). Report null/mixed results plainly.

## Full brief (verbatim)

> You are my autonomous research engineer in Claude Code. Run a mechanistic-interpretability experiment end to end, explain your choices so that I learn (this is my first hands-on interp project; I'm working through ARENA), and produce outputs I can turn into a blog post.
>
> HOW TO WORK
> - First write CLAUDE.md with this full brief, then a task list. Tick tasks off as you go.
> - Ask me only for things only I can provide, in a "NEED FROM MURT" block with numbered steps, then keep working on anything else. Secrets go in .env only and are never committed.
> - Run long jobs in the background with logs and checkpoints, and cache activations to disk.
> - Update STATUS.md after each stage and print a short summary. Commit after each stage; push if I give a GitHub URL.
> - Verify by running. Never fabricate numbers: every number must trace to a file in results/.
>
> RESEARCH QUESTION
> "Does an LLM have a linear 'emotional tone' direction that causally controls its outputs, or does a contrastive probe mostly detect the instruction text used to create the contrast?" Treat this as an open question. Either outcome is a valid result.
>
> PRIOR WORK (cite accurately; don't invent details): Zou et al. 2023 "Representation Engineering"; Turner et al. 2023 "Activation Addition"; Rimsky et al. 2024 "Contrastive Activation Addition"; Hewitt & Liang 2019 (probe control tasks).
>
> COMPUTE AND MODELS
> - Local prototyping on my RTX 5060 laptop GPU (8 GB VRAM, Windows): gemma-2-2b-it or Qwen2.5-1.5B/3B-Instruct, to debug hooks and maths at no cost.
> - Main runs on Modal (credentials via `modal setup`; ask me): Llama-3.1-8B-Instruct in fp16 on an A10G/L4. Use TransformerLens, nnsight or HF hooks. Hard cap $10 Modal.
> - Judge on Fireworks (FIREWORKS_API_KEY in .env; ask me): a large model from the live serverless list (prefer gpt-oss-120b), temperature 0. Hard cap $8.
> - The 2B/3B model doubles as a free replication.
>
> DATASETS
> D1 (instruction contrast, as in the original design): about 150 neutral tasks ("Describe a house", "Explain photosynthesis", "Write a Python function to sort a list") × two system prompts: EMPATHETIC vs COLD/ROBOTIC. Write 5 paraphrases of each system prompt and rotate them.
> D2 (content contrast, NO instructions): about 300 short user messages that are emotionally charged (grief, joy, fear, anger) vs emotionally neutral, matched on topic and length. No system prompt about tone.
> D3 (format shift): the same D1/D2 contrasts in other formats: multiple choice, JSON output requested, and a different chat template or plain completion style.
> Split everything into train/test by task or topic (no topic appears in both). Check by script that there's no leakage.
>
> STAGE 1: READING (probes)
> - Residual stream at every layer, at the final prompt token.
> - Direction: the mean difference (emotional minus neutral) per layer on the D1 train split. Also train an L2 logistic probe per layer.
> - Report per layer, with bootstrap 95% CIs:
>   a. Accuracy on held-out D1.
>   b. TRANSFER: the D1-trained direction or probe applied to D2 (emotion in the content, no instructions). This is the key test of whether it encodes emotion or instruction text.
>   c. Transfer to D3 formats.
>   d. Hewitt & Liang control task (random labels) and selectivity.
>   e. Confound probe: a probe that only detects which system-prompt PARAPHRASE was used, or the presence of the tone words. How much of the D1 accuracy can this explain?
>   f. A cosine similarity matrix between the D1 direction and a D2-trained direction, per layer.
> - DECONFOUNDING (exploratory): estimate a "format/instruction" subspace (e.g. by SVD of directions that separate formats or paraphrase sets), project it out of the D1 direction, and re-test transfer to D2. Does transfer improve?
>
> STAGE 2: CONTROL (steering)
> - Choose the steering layer(s) from the Stage 1 transfer results, not D1 accuracy alone. Explain the choice.
> - On about 60 held-out neutral prompts (none used before), add c·v to the residual stream at all positions, for c in {−8, −4, −2, 0, 2, 4, 8} × (the norm of the typical residual), or a similar scale; calibrate the scale first.
> - Compare three steering vectors: (i) the D1 direction; (ii) the D2 direction; (iii) the D1 direction with the format subspace projected out (if computed); plus (iv) a random direction with the same norm as the control.
> - Measure, per condition:
>   - The judge's emotional-intensity score (1–10, frozen rubric, blind to condition, order randomised), plus a judge score for task completion (did it still answer the task?).
>   - Perplexity of the steered output, measured by the unsteered model, as a fluency check.
>   - Side effects (exploratory): rates of refusal, apologies and bullet lists (regex plus judge).
> - Validate the judge: I will hand-label 40 outputs; ask me in a NEED FROM MURT block, and give me a simple CSV to fill in. Report the judge–human agreement.
>
> PRE-REGISTRATION
> Before the main runs, write docs/PREREG.md: hypotheses (H1: the D1 probe is accurate in mid-to-late layers; H2: the D1 direction transfers to D2 above chance; H3: steering with the D1 direction raises judge emotion scores relative to the random direction without collapsing task completion; H4: the D2 direction steers at least as well as the D1 direction), metrics, thresholds and analysis. Label everything after that as exploratory.
>
> UNIT TESTS before paid runs: hook shapes; token position; label alignment; no train/test leakage; steering with c = 0 reproduces the baseline outputs exactly (greedy decoding).
>
> REQUIRED OUTPUTS (hand these back to me when finished)
> 1. docs/SUMMARY.md (15 lines max): H1–H4 verdicts with numbers and CIs; the best layer; the D1→D2 transfer result (the headline); confound-probe and control-task results; the effect of deconfounding; the steering dose-response for each vector vs the random control; task-completion and perplexity costs; the side effects found; whether the small model agrees; and a one-line honest verdict ("emotion direction", "mostly instruction detector", or "mixed").
> 2. Figures (PNG and PDF, 300 dpi, readable):
>    F1 probe/direction accuracy vs layer: D1 held-out, D1→D2 transfer, D1→D3, and the control task;
>    F2 cosine similarity of the D1 vs D2 directions per layer;
>    F3 steering dose-response: judge emotion score vs c for each vector plus random, with CIs;
>    F4 fluency/task cost: perplexity and task completion vs c;
>    F5 a qualitative panel of 3 prompts × (−c, 0, +c) outputs, selected by a fixed rule (e.g. the first 3 test prompts), not cherry-picked.
> 3. results/RESULTS.md generated by a script; raw JSON metrics; all generations saved as JSONL.
> 4. A plain-language README covering the method, results and limitations: probe decodability is not the same as the model "using" the concept; one judge; small datasets; steering scale choices; judge bias.
>
> RULES: No overclaiming ("proves", "sociopathic", "brain"). Report null or mixed results plainly. Simple language in the README.
>
> Start now: write CLAUDE.md and the task list, build the datasets and the local prototype on the small model, run the unit tests, then ask me for the Modal, Hugging Face and Fireworks credentials in a NEED FROM MURT block.

Extra context from Murt (chat): GitHub repo https://github.com/MurtuzaShaikh26/emotion-probe ;
Modal profile `ronitk964` (token set via `modal token set`, stored in `~/.modal.toml`, not in repo);
"Please ensure you don't waste the Modal credits."

## Working decisions (engineering; scientific choices are frozen in docs/PREREG.md)
- **Models.** Local: `Qwen/Qwen2.5-3B-Instruct` (ungated, fp16, fits 8 GB) for prototyping and as the
  replication model. Main: `unsloth/Meta-Llama-3.1-8B-Instruct` — an ungated byte-identical mirror of
  `meta-llama/Llama-3.1-8B-Instruct`, so no Hugging Face token is needed.
- **Hooks.** Plain Hugging Face forward hooks on `model.model.layers[i]` (no TransformerLens dependency,
  which keeps the Modal image small). "Layer ℓ residual" = output of decoder block ℓ (ℓ = 0..L-1);
  we also store the embedding output as ℓ = -1 → stored index 0 (so arrays have L+1 rows).
- **Token position.** Final prompt token after the chat template's generation prompt
  (e.g. Llama: the `\n\n` after `<|start_header_id|>assistant<|end_header_id|>`). Left padding in batches.
- **Modal frugality.** One A10G container per job, weights cached in a Modal Volume, all D1/D2/D3
  activations extracted in a single call; steering generations in one batched call; `timeout` set on
  every function; nothing runs until the local pipeline passes all tests on Qwen.
- Code layout: `src/emoprobe/` (library), `scripts/` (entry points), `tests/`, `data/` (built datasets),
  `cache/` (activations, gitignored), `results/` (metrics JSON, generations JSONL, RESULTS.md),
  `figures/`, `docs/`.
