"""Assemble mammqa_hop_trace_300.ipynb from nb_src.py + worker_src.py + official prompts.py + frozen selection."""
import json, re, sys, hashlib, runpy
from pathlib import Path

HERE = Path(__file__).parent
PROMPTS = Path(sys.argv[1])                 # MAMMQA-main/prompts.py
SEL = json.loads((HERE / "selected_questions_300.json").read_text(encoding="utf-8"))

ns = runpy.run_path(str(PROMPTS))
fp = {k: hashlib.sha256(ns[v].encode()).hexdigest()[:16] for k, v in
      {"stage1": "Agent_stage1_system_prompt", "stage2": "Agent_stage2_system_prompt", "stage3": "Agent_stage3_system_prompt"}.items()}
worker = (HERE / "worker_src.py").read_text(encoding="utf-8").replace(
    "# __OFFICIAL_PROMPTS__", PROMPTS.read_text(encoding="utf-8").rstrip())
(HERE / "mammqa_trace_worker.py").write_text(worker, encoding="utf-8")
compile(worker, "worker", "exec")

src = (HERE / "nb_src.py").read_text(encoding="utf-8")
src = src.replace("__SELECTION__", json.dumps(SEL, ensure_ascii=False)).replace("__FINGERPRINTS__", json.dumps(fp))
src = src.replace("__SEL_SHA__", SEL["sha256_questions"][:16])
WORKER_MAGIC = "%%writefile /kaggle/working/mammqa_trace_worker.py\n"

cells = []
for chunk in re.split(r"(?m)^# %%", src):
    if not chunk.strip():
        continue
    if chunk.startswith(" [markdown]"):
        body = chunk[len(" [markdown]"):].strip("\n")
        md = "\n".join(l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l) for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
        continue
    code = chunk.strip("\n")
    if "# __WORKER_CELL__" in code:
        code = WORKER_MAGIC + "# Cell 7 — worker script (one process per GPU); written to disk, launched by Cell 8\n" + worker.rstrip("\n")
    else:
        compile(code, "cell", "exec")
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": code.splitlines(keepends=True)})
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                              "dataSources": [{"sourceType": "datasetVersion", "datasetSlug": "multimodalqa-dataset", "ownerSlug": "adibraian"},
                                              {"sourceType": "datasetVersion", "datasetSlug": "final-dataset-images-mmqa", "ownerSlug": "adibraian"}]}}}
out = HERE / "mammqa_hop_trace_300.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("cells:", len(cells), "| prompt fingerprints:", fp, "| bytes:", out.stat().st_size)
