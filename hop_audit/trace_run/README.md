# trace_run: 300-question MAMMQA hop-trace run (Kaggle)

- `PREREGISTRATION.md`: the claim, the decision rule and the robustness checks, fixed before the run.
- `selected_questions_300.json`: the frozen sample (sha256 of the question list `680d34ad38ea7d7a…`).
- `mammqa_hop_trace_300.ipynb`: the Kaggle notebook. Settings: GPU T4 x2, Internet on, inputs
  `adibraian/multimodalqa-dataset` and `adibraian/final-dataset-images-mmqa`. Use Save & Run All.
- `src/`: the notebook's sources. `build.py` assembles the notebook from `nb_src.py`, `worker_src.py`,
  the official `prompts.py` and the selection. `run_nb_mock.py` runs it locally with a mock model.

After the run, download the output and audit it:

    python ..\run_hop_audit.py --source mammqa_300=mammqa_official:<download>\mammqa_trace\predictions\mammqa_predictions.jsonl --out ..\outputs_300

If the run stops before all 300 are done, add the notebook's own output as an input and run it again.
Finished questions are skipped.
