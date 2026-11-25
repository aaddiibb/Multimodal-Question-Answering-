# Pre-registration: hop-trace run (300 MMQA dev questions, released MAMMQA prompts)

Written 2026-10-30 (Asia/Dhaka), **before** any model output for these questions exists.
The selection file `selected_questions_300.json` (questions sha256 `680d34ad38ea7d7a…`) and the hop_audit
code version (`hop_audit/` at this commit, 25 tests passing) are frozen with this document.

## Claim under test

> In MAMMQA, most wrong answers to compositional questions are **not** caused by missing or lost
> bridge information. The bridge (or the answer itself) is already present inside the pipeline, and
> the failure happens at composition, commitment or aggregation.

## Setup (fixed)

- **Questions:** 300 MMQA dev questions with at least one latent gold hop. They are sampled in
  proportion to type (Compose 192, Compare 49, TableQ 33, Intersect 26), seed 20261030, and exclude
  the 52 questions inspected in earlier pilots. They are processed in shuffled order, so any finished
  prefix is a random subsample.
- **Pipeline:** MAMMQA as released. The prompts are `prompts.py` verbatim (checked by sha256), the
  messages follow `agents.py` exactly (7 agents), and the context is the paper's full per-question
  context: `metadata.text_doc_ids`, `table_id` and `image_doc_ids`, gold plus distractors, with
  official image pixels and **no image titles**.
- **One deviation**, the same as in all earlier pilots: the aggregator is asked to end with
  `<answer>…</answer>` so that EM is defined.
- **Model:** Qwen2.5-VL-7B-Instruct, 4-bit NF4, greedy decoding, repetition penalty 1.05.
  Max new tokens: 448 (Stage I), 512 (Stage II), 320 (Stage III). Visual budget: 3,200 tokens per call.
- **No gold field enters any prompt.** Predictions are frozen and hashed before gold is read.
- **Scoring:** `hop_audit` exactly as frozen, including the `aliases.json` entries existing at freeze
  time. Aliases added afterwards are reported separately, as a sensitivity analysis.

## Primary analysis

Let W be the wrong answers (EM = 0), excluding SURFACE_FORM (content right, metric strict).

**P = (COMPOSITION_FAILURE + AGGREGATION_LOSS + COMMITMENT_FAILURE) / |W|**

- **Supported** if the 95% bootstrap CI for P (10,000 resamples over questions) lies entirely above 0.50.
- **Refuted** if it lies entirely below 0.50.
- **Inconclusive** otherwise.

The minimum usable sample is 200 completed questions. Below that, results are reported as
descriptive only.

## Secondary analyses (descriptive, with bootstrap CIs)

1. The full label distribution, overall and by question type.
2. Bridge recall at Stage I, at Stage II, and in the final text, overall and by hop modality.
   Expectation: image hops are lowest, because no titles are given.
3. Strict hand-off drop rate: X-anchored Stage II dropping a hop that only Stage I-X could pass on.
4. Abstention: how often the final answer abstains in COMPOSITION_FAILURE cases.
5. EM by type, against the pilots. This is context only, not a claim.

## Robustness checks (decided now)

- **R1.** Fuzzy matching off.
- **R2.** No aliases.
- **R3.** All golds treated as strong, i.e. the weak-gold guard switched off. Shows how much the
  guard matters.
- **R4. Manual validation:** 60 questions, drawn at random with seed 7 among those that have a label
  other than CORRECT. Each is checked by reading the traces. Report agreement with the automatic label
  and the confusion matrix. If agreement on the "information present" vs "absent" split is below 80%,
  the primary result is reported as unreliable.
- **R5. Mention vs use:** in the R4 sample, how often a hop counted as REACHED was explicitly rejected
  or misused by the agent.

## What would change the plan

- **P supported:** the paper's contribution is the diagnosis plus an intervention aimed at
  composition and commitment (not at retrieval or perception).
- **P refuted:** the bottleneck is extraction and hand-off, so the intervention targets carrying
  bridges between stages.
- **Inconclusive:** extend to 600 questions with the same protocol before deciding.
