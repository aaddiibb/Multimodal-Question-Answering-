# MAMMQA Qwen Full Pipeline

This bundle rebuilds the MAMMQA pipeline with Qwen instead of OpenAI GPT.
It uses Qwen through DashScope's OpenAI-compatible API, so the Python class is
`openai.OpenAI`, but the API key is `DASHSCOPE_API_KEY` and the `base_url` points
to DashScope/Qwen. No OpenAI API key is used.

## What is included

- `mammqa_qwen_full_pipeline.ipynb` - self-contained notebook. It writes the modules below and runs experiments.
- `qwen_client.py` - DashScope/Qwen client, image handling, retry logic.
- `Dataloader.py` - `MultiModalQADataLoader` compatible with MMQA JSON/JSONL files.
- `agents_qwen.py` - zero-shot, CoT, and full 3-stage MAMMQA agents.
- `treeofthoughts_qwen.py` and `tot_dfs_qwen.py` - Tree-of-Thoughts candidate generation, scoring, and DFS pruning.
- `Eval.py` - EM, token F1, and lemma-match evaluation.
- `run_agents_qwen.py`, `run_baselines_qwen.py`, `run_tot_qwen.py` - CLI runners.

## Install

```bash
pip install -U openai python-dotenv pandas tabulate requests pillow spacy
python -m spacy download en_core_web_sm  # optional; Eval.py falls back if unavailable
```

## Environment

Set one secret/key:

```bash
export DASHSCOPE_API_KEY="your_dashscope_key"
```

Optional model/region settings:

```bash
export QWEN_REGION="intl"              # intl, us, or beijing
export QWEN_TEXT_MODEL="qwen-plus"     # e.g. qwen-plus, qwen-turbo
export QWEN_VISION_MODEL="qwen3-vl-plus"
```

If your API key was created for a China-region DashScope account, use:

```bash
export QWEN_REGION="beijing"
```

## Run full MAMMQA agents

```bash
python run_agents_qwen.py \
  --root /kaggle/input/datasets/adibraian/multimodalqa-dataset \
  --n 3 \
  --out outputs/qwen_mammqa_predictions.jsonl
```

Cost-saving mode that runs only the modality implied by the MMQA question type:

```bash
python run_agents_qwen.py --n 3 --reduce-modalities-by-type
```

## Run baselines

```bash
python run_baselines_qwen.py --method zs  --n 3 --out outputs/qwen_zs.jsonl
python run_baselines_qwen.py --method cot --n 3 --out outputs/qwen_cot.jsonl
```

## Run ToT

```bash
python run_tot_qwen.py --idx 0 --k 3 --max-depth 3 --out outputs/qwen_tot.jsonl
```

## Evaluate

```bash
python Eval.py --pred outputs/qwen_mammqa_predictions.jsonl --out outputs/qwen_mammqa_eval.json
```
