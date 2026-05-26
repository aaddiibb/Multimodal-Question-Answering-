
# MAMMQA Kaggle Local Complete Pipeline — No API Keys

This bundle contains a complete Kaggle notebook:

- `mammqa_kaggle_local_complete_no_api.ipynb`

It uses the dataset path from the previous notebooks:

```text
/kaggle/input/datasets/adibraian/multimodalqa-dataset
```

and automatically resolves:

```text
MMQA_dev.jsonl
MMQA_texts.jsonl
MMQA_tables.jsonl
MMQA_images.jsonl
```

## What it includes

- Local Qwen text/table/reasoning model
- Local SmolVLM image-evidence model
- Stage 1 modality specialist agents
- Stage 2 cross-modal refinement agents
- Stage 3 aggregator
- Zero-shot baseline
- CoT baseline
- Optional lightweight Tree-of-Thoughts
- CSV and JSONL saving
- EM and token-F1 evaluation

## No API keys

This notebook does not use:

- OpenAI API key
- DashScope API key
- Gemini key
- any paid API key

It downloads open-source Hugging Face models into the Kaggle runtime.

## Kaggle setup

1. Attach the MultimodalQA Kaggle dataset.
2. Enable GPU.
3. Enable Internet for first run.
4. Run all cells.

## First-run recommended settings

```python
RUN_N_EXAMPLES = 3
TEXT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
USE_IMAGE_AGENT = True
REDUCE_MODALITIES_BY_QUESTION_TYPE = True
STRICT_PAPER_MODE = False
RUN_BASELINES = True
RUN_TOT = False
```

If memory fails, use:

```python
TEXT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
USE_IMAGE_AGENT = False
```

## Outputs

Saved in:

```text
/kaggle/working/mammqa_local_complete_outputs/
```

Main files:

```text
mammqa_local_predictions.jsonl
mammqa_local_predictions.csv
local_baselines.jsonl
local_baselines.csv
local_tot.jsonl
local_tot.csv
```
