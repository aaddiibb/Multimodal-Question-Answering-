# hop_audit: hop-level audit of multi-agent MMQA pipelines

This folder holds the code for the paper plan "Hop-Level Audit of Multi-Agent MMQA". It answers
**where** a multi-agent pipeline loses the information a compositional question needs. It does not
only report whether the final answer is right.

MultiModalQA annotates every compositional dev question with gold **intermediate answers**: the
bridge entity of a Compose question, the two items of a Compare, the candidate sets of an Intersect.
Each one is tagged with its modality. The audit traces each gold hop through every agent's output and
gives every wrong answer one label: the earliest point of failure.

No GPU and no new model calls are needed. It reads traces that already exist.

## Run it

```
cd "E:\Multimodal Question Answering\hop_audit"
python run_hop_audit.py                 # audits the 4 trace sets already in this project
python run_hop_audit.py --dev-stats     # + gold hop statistics over all 2441 dev questions
python -m unittest discover -s tests -v # 25 tests (unit + integration on the CV pilot)
```

It uses only the Python standard library (3.8+). If scipy is installed it is used for answer
alignment; otherwise an exact standard-library fallback is used. The two agree on 3,000 random cases.

Audit a new run:

```
python run_hop_audit.py --source my_run=mammqa_official:path\to\mammqa_predictions.jsonl
python run_hop_audit.py --source pilot=cv_pilot:path\to\predictions_shared.jsonl,path\to\predictions_mammqa.jsonl
```

Readers: `cv_pilot`, `mammqa_official`, `hm_oracle`. To support another pipeline, add a reader in
`hop_audit/traces.py`. A reader turns one output row into agents with `stage`, `answer`, `text`,
`raw_inputs` (the modalities the agent saw raw) and `handoff_from` (the upstream agents whose output
it received).

## What is measured

**Gold hops** (`hop_audit/gold.py`). Each set in `metadata.intermediate_answers` is one hop.

- A hop item already named in the question is **given**, for example "Nightwing" in "Nightwing or
  the movie with ...". Only **latent** hops count. On dev, 330 of the 1,579 hops are given.
- For Intersect sets, only the item(s) that overlap the final answer are required.

**Per hop and per agent** (`hop_audit/scoring.py`):

- *available*: the agent saw the hop's modality as raw evidence.
- *surfaced*: the agent's output mentions every required hop item. Matching uses normalized
  whole-token matching, accent folding, number words, disambiguator stripping (`Mask (1985 film)` →
  `Mask`), fuzzy matching for names of 3 or more tokens, and the reviewed aliases in `aliases.json`.
- *bridge recall by stage*: the share of latent hops surfaced by at least one agent at that stage.
- *stage-level hand-off loss*: a hop surfaced at Stage I and missing from every agent of the last
  synthesis stage.
- *strict hand-off loss*: in MAMMQA, the cross-modal agent anchored on modality X never sees X raw,
  only X's Stage-I insight. When a hop lives in X and Stage I surfaced it, the downstream agent either
  **carried** it or **dropped** it. This is exact, and it is computed only where the information flow
  is known (the CV pilot and official MAMMQA, not HM-RAG).

**One label per wrong answer**, checked in this order:

| label | meaning |
|---|---|
| SURFACE_FORM | the final answer contains every gold item, but official EM is 0 |
| AGGREGATION_LOSS | an intermediate agent committed to the gold answer; the final answer did not |
| COMMITMENT_FAILURE | the gold answer appears in some agent's reasoning, but no agent committed to it |
| NO_EVIDENCE | a needed hop or the answer is in a modality that was not in the context |
| EXTRACTION_FAILURE | a needed hop was available, but no agent surfaced it |
| HANDOFF_LOSS | a hop surfaced early, but no agent at the last synthesis stage carried it |
| COMPOSITION_FAILURE | every latent hop reached the last synthesis stage, but no agent produced the answer |

Guards against false credit:

- An answer field counts as a *commitment* only if it is not an abstention. This includes sentences
  such as "There is insufficient information…".
- Weak golds (numbers, yes/no, and single lowercase nouns like "horses") count only in short answers,
  never as reasoning mentions.
- In Compare questions whose answer is itself a hop item, a long sentence that identifies the item is
  treated as hop extraction, not as the answer.
- Mentions that were deliberately not scored are listed per question (`unscored_gold_mentions`).
- Near misses (e.g. "Jim Caviezel" vs "James Patrick Caviezel") are listed for alias review in each
  `summary.json`. They never change a label until a reviewed alias is added to `aliases.json`.

The official MMQA EM/F1 (DROP-style bags + optimal alignment) is used for correctness. It reproduces
the EM recorded by every source run.

## Outputs (`outputs/`)

- `report.md`: all pipelines side by side.
- `manifest.json`: sha256 of every input, alias count and dev statistics.
- `<pipeline>/per_question.jsonl`: full detail, including per-agent surfacing, channels and flags.
- `<pipeline>/per_question.csv`, `per_hop.csv`: the same data, flattened for Excel.
- `<pipeline>/summary.json`: recall by stage, hand-off loss, label counts and the near-miss list.

## Current results (code check only, not claims)

| pipeline | n | EM | latent hops | strict hand-off drops | dominant error label |
|---|---|---|---|---|---|
| CV pilot, baseline aggregator | 16 | 0.19 | 15 | 2 of 7 | COMPOSITION_FAILURE (6 of 13) |
| CV pilot, CV aggregator | 16 | 0.19 | 15 | 2 of 7 | same labels as baseline |
| HM-RAG oracle | 16 | 0.25 | 16 | n/a | COMPOSITION_FAILURE (5 of 12) |
| MAMMQA released prompts | 20 | 0.25 | 12 | 4 of 7 | COMPOSITION_FAILURE (6 of 15) |

Two patterns are worth testing at scale.

1. In all three pipelines, COMPOSITION_FAILURE is the largest error class. Counting aggregation and
   commitment failures as well, 9 of 13 (pilot), 8 of 12 (HM-RAG) and 9 of 15 (MAMMQA) errors happened
   after every bridge, or the answer itself, was already inside the pipeline. The agents then abstained
   or composed the wrong answer. Missing evidence was not the main problem.
2. The CV (claim-verification) aggregator changed no label. Every error it could act on arose
   upstream of it, which is consistent with the pilot's 18.75% vs 18.75% EM.

With 16–20 questions, one question is 5–6 points. These numbers check the code and do not support
claims.

## Known limits

- Mention is not the same as correct use: an agent can name the bridge and then reject it. The
  per-agent `answer` fields separate commitment from mention, but reasoning quality is not judged.
- In the CV pilot, images were shown together with their Wikipedia titles. That makes image-hop
  "extraction" partly string copying, so read its image-hop recall as an upper bound.
- `mammqa_full_outputs` had no images, so its image hops are NO_EVIDENCE by construction.
- HM-RAG does not log the cross-modal agents' raw inputs, so strict hand-off loss is not computed for it.

## Files

```
run_hop_audit.py        command-line entry point
aliases.json            reviewed aliases (mention checks only; never EM)
hop_audit/gold.py       gold hop structure from MMQA dev
hop_audit/matching.py   normalisation, mention matching, official EM/F1
hop_audit/traces.py     readers -> common trace format
hop_audit/scoring.py    hop scoring, channels, failure attribution, summaries
hop_audit/report.py     CSV/JSON/Markdown writers
tests/test_hop_audit.py unit + integration tests
```

Inputs used here: `cv_mammqa_16/` (Kaggle debate-mammqa v358107244; `predictions_shared.jsonl` sha256
`f17b8d46…`, matching the run's PREDICTIONS_FROZEN.json), `mammqa_full_outputs/`, and the HM-RAG smoke
run under `New folder/`.
