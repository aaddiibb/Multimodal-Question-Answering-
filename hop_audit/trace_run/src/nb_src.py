# %% [markdown]
# # MAMMQA hop-trace run: 300 MMQA dev questions, released prompts
#
# This notebook produces full per-agent traces of **MAMMQA as released** on 300 MultiModalQA dev questions,
# so that `hop_audit` can test the pre-registered claim: *most wrong answers happen after the bridge
# information is already inside the pipeline*. See `PREREGISTRATION.md` in the project's `hop_audit` folder.
#
# | | |
# |---|---|
# | **Questions** | 300 dev questions with at least one latent gold hop. Frozen list (sha256 `__SEL_SHA__…`), stratified by type, shuffled order. |
# | **Pipeline** | `prompts.py` verbatim (sha256 checked), `agents.py` message formats, 7 agents. The paper's per-question context (`text_doc_ids`, `table_id`, `image_doc_ids`) with official pixels and **no image titles**. |
# | **Only deviation** | The aggregator ends with `<answer>…</answer>`, so EM is defined. |
# | **Model** | Qwen2.5-VL-7B-Instruct, 4-bit NF4, greedy. **One full copy per T4**, two workers in parallel. |
# | **Leakage** | Prompts are built from gold-free items. Predictions are frozen and hashed before gold is read. |
#
# **Settings:** accelerator **GPU T4 x2**, Internet **on**, inputs `adibraian/multimodalqa-dataset` and
# `final-dataset-images-mmqa`. Use **Save & Run All** (a background commit, up to 12 h).
#
# **Resume:** if the run stops before 300, add this notebook's own output as an input and run it again.
# Finished questions are skipped.

# %%
# Cell 1 — environment
import os, sys, subprocess, importlib, json, re, time, math, random, hashlib, gzip, zipfile, string, shutil, glob
from pathlib import Path
from collections import Counter, defaultdict
NOTEBOOK_START = time.time()
MOCK = os.environ.get("TRACE_MOCK", "0") == "1"          # local plumbing test only; never set on Kaggle
if not MOCK:
    if importlib.util.find_spec("bitsandbytes") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.45"], check=False)
    import torch
    N_GPU = torch.cuda.device_count()
    print("PyTorch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPU)])
    import transformers; print("transformers", transformers.__version__)
else:
    N_GPU = int(os.environ.get("TRACE_MOCK_GPUS", "2"))
print("MOCK =", MOCK, "| workers =", N_GPU)

# %%
# Cell 2 — configuration (frozen; edit only MAX_RUNTIME_HOURS or MODEL_PATH_OVERRIDE)
INPUT_ROOT = Path(os.environ.get("TRACE_INPUT_ROOT", "/kaggle/input"))
OUTPUT_ROOT = Path(os.environ.get("TRACE_OUTPUT_ROOT", "/kaggle/working/mammqa_trace"))
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH_OVERRIDE = None
MAX_RUNTIME_HOURS = float(os.environ.get("TRACE_MAX_HOURS", "11.0"))   # stop STARTING questions after this
SELECTION = __SELECTION__
assert hashlib.sha256(json.dumps(SELECTION["questions"], sort_keys=True).encode()).hexdigest() == SELECTION["sha256_questions"]
ORDER = [q["qid"] for q in SELECTION["questions"]]
N_LIMIT = int(os.environ.get("TRACE_N", str(len(ORDER))))
ORDER = ORDER[:N_LIMIT]
OFFICIAL_PROMPT_FINGERPRINTS = __FINGERPRINTS__
DIRS = {k: OUTPUT_ROOT / k for k in ["config", "inputs", "cache", "checkpoints", "errors", "logs", "predictions", "metrics", "images"]}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(DIRS["logs"] / "main.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(path)

log(f"{len(ORDER)} questions | selection sha256 {SELECTION['sha256_questions'][:16]} | allocation {SELECTION['allocation']}")

# %%
# Cell 3 — locate MMQA files and official image pixels
def walk_files(root):
    for dp, _, fns in os.walk(root, followlinks=True):
        for n in fns:
            yield Path(dp) / n

def read_jsonl(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

ALL_INPUT_FILES = [p for p in walk_files(INPUT_ROOT)]
def find_mmqa(kind):
    hits = sorted([p for p in ALL_INPUT_FILES if f"MMQA_{kind}" in p.name and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))],
                  key=lambda p: (p.name.endswith(".gz"), len(str(p))))
    if not hits:
        raise FileNotFoundError(f"MMQA_{kind} not found under {INPUT_ROOT}: attach adibraian/multimodalqa-dataset")
    return hits[0]
MMQA = {k: find_mmqa(k) for k in ["dev", "texts", "tables", "images"]}
for k, v in MMQA.items():
    log(f"{k:<7} {v}")

# %%
# Cell 4 — build GOLD-FREE inference items from the paper's haystack, then drop the gold-bearing rows
from dataclasses import dataclass
dev_rows = {r["qid"]: r for r in read_jsonl(MMQA["dev"]) if r["qid"] in set(ORDER)}
assert len(dev_rows) == len(ORDER), "selected qid missing from dev"
need_text = {t for r in dev_rows.values() for t in r["metadata"].get("text_doc_ids", [])}
need_table = {r["metadata"].get("table_id") for r in dev_rows.values()} - {None}
need_image = {i for r in dev_rows.values() for i in r["metadata"].get("image_doc_ids", [])}
TEXTS = {r["id"]: r for r in read_jsonl(MMQA["texts"]) if r["id"] in need_text}
TABLES = {r["id"]: r for r in read_jsonl(MMQA["tables"]) if r["id"] in need_table}
IMAGES_META = {r["id"]: r for r in read_jsonl(MMQA["images"]) if r["id"] in need_image}
log(f"records: texts {len(TEXTS)}/{len(need_text)} tables {len(TABLES)}/{len(need_table)} images {len(IMAGES_META)}/{len(need_image)}")

IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
needed_names = {Path(str(m.get("path", ""))).name for m in IMAGES_META.values() if m.get("path")}
IMAGE_FILES, stems = {}, {Path(n).stem: n for n in needed_names}
for p in ALL_INPUT_FILES:
    if p.suffix.lower() in IMG_EXT:
        if p.name in needed_names and p.name not in IMAGE_FILES:
            IMAGE_FILES[p.name] = p
        elif p.stem in stems and stems[p.stem] not in IMAGE_FILES:
            IMAGE_FILES[stems[p.stem]] = p
missing = needed_names - set(IMAGE_FILES)
for z in [p for p in ALL_INPUT_FILES if p.suffix.lower() == ".zip"] if missing else []:
    try:
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                n = Path(member).name
                if n in missing:
                    out = DIRS["images"] / n
                    if not out.exists():
                        out.write_bytes(zf.read(member))
                    IMAGE_FILES[n] = out
    except zipfile.BadZipFile:
        continue
    missing = needed_names - set(IMAGE_FILES)
    if not missing:
        break
log(f"official image files resolved: {len(IMAGE_FILES)}/{len(needed_names)}")

def unique_headers(h):
    seen, out = Counter(), []
    for x in h:
        out.append(x if seen[x] == 0 else f"{x}_{seen[x]}"); seen[x] += 1
    return out

def table_markdown(rec):
    t = rec.get("table", {})
    headers = [h.get("column_name", "") for h in t.get("header", [])]
    rows = [[c.get("text", "") for c in row] for row in t.get("table_rows", [])]
    width = max([len(headers)] + [len(r) for r in rows] + [0])
    if not headers or len(headers) != (len(rows[0]) if rows else len(headers)):
        headers = [str(i) for i in range(width)]
    headers = unique_headers(headers)
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ").strip()
    lines = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join(":---" for _ in headers) + "|"]
    for r in rows:
        r = (r + [""] * len(headers))[:len(headers)]
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)

ITEMS = {}
for qid in ORDER:
    meta = dev_rows[qid]["metadata"]
    texts = [TEXTS[t] for t in meta.get("text_doc_ids", []) if t in TEXTS]
    tab = TABLES.get(meta.get("table_id"))
    paths, miss = [], []
    for iid in meta.get("image_doc_ids", []):
        rec = IMAGES_META.get(iid)
        p = IMAGE_FILES.get(Path(str(rec.get("path", ""))).name) if rec else None
        (paths.append(str(p)) if p else miss.append(iid))
    ITEMS[qid] = {   # the ONLY fields a prompt can see
        "qid": qid, "question": dev_rows[qid]["question"],
        "text_block": "\n".join(f"title: {t.get('title', 'No Title')}, text: {t.get('text', 'No Text')}" for t in texts),
        "table_block": (f"Table title: {tab.get('title', 'No Title')}\n" + table_markdown(tab)) if tab else "No table data",
        "has_table": tab is not None, "image_paths": paths, "missing_images": miss,
    }
FORBIDDEN = {"answers", "metadata", "supporting_context", "intermediate_answers", "type", "modalities"}
assert all(not (set(v) & FORBIDDEN) for v in ITEMS.values())
atomic_json(DIRS["inputs"] / "items.json", ITEMS)
n_img = [len(v["image_paths"]) for v in ITEMS.values()]
log(f"items written | images/question mean {sum(n_img)/len(n_img):.1f} max {max(n_img)} | questions with missing image files: {sum(bool(v['missing_images']) for v in ITEMS.values())}")
del dev_rows                                   # gold-bearing rows are not kept during inference

# %%
# Cell 5 — resume: import checkpoints and call cache from an attached earlier version of this notebook
imported = 0
for ck_dir in [p for p in INPUT_ROOT.rglob("checkpoints") if p.is_dir() and (p.parent / "inputs" / "items.json").exists()]:
    for f in ck_dir.glob("*.json"):
        dst = DIRS["checkpoints"] / f.name
        if f.stem in ITEMS and not dst.exists():
            shutil.copy2(f, dst); imported += 1
    for f in (ck_dir.parent / "cache").glob("*.json"):
        dst = DIRS["cache"] / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
log(f"resume: imported {imported} finished questions from earlier versions")

# %%
# Cell 6 — model source (download ONCE before workers start) and run config
def resolve_model_source():
    if MODEL_PATH_OVERRIDE:
        return str(MODEL_PATH_OVERRIDE)
    for cfg in INPUT_ROOT.rglob("config.json"):
        s = str(cfg.parent).lower()
        if "qwen" in s and "vl" in s and "7b" in s:
            try:
                if json.loads(cfg.read_text()).get("model_type") == "qwen2_5_vl":
                    return str(cfg.parent)
            except Exception:
                pass
    if MOCK:
        return MODEL_ID
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja"])

MODEL_SOURCE = resolve_model_source()
log(f"model source: {MODEL_SOURCE}")
RUN_CONFIG = {
    "MOCK": MOCK, "MODEL_ID": MODEL_ID, "MODEL_SOURCE": MODEL_SOURCE, "ORDER": ORDER,
    "OFFICIAL_PROMPT_FINGERPRINTS": OFFICIAL_PROMPT_FINGERPRINTS,
    "MAX_NEW_TOKENS": {"stage1": 448, "stage2": 512, "stage3": 320}, "MAX_INPUT_TOKENS": 14000,
    "VISUAL_TOKEN_BUDGET": 3200, "MIN_IMAGE_TOKENS": 64, "MAX_IMAGE_TOKENS": 1024,
    "DECODING": "greedy", "REPETITION_PENALTY": 1.05,
    "DEADLINE_EPOCH": NOTEBOOK_START + MAX_RUNTIME_HOURS * 3600,
    "MOCK_SLEEP": float(os.environ.get("TRACE_MOCK_SLEEP", "0")),
    "selection_sha256": SELECTION["sha256_questions"],
}
atomic_json(DIRS["config"] / "run_config.json", RUN_CONFIG)
atomic_json(DIRS["config"] / "selected_questions.json", SELECTION)

# %%
# Cell 7 — the worker script (one process per GPU)
# __WORKER_CELL__

# %%
# Cell 8 — launch one worker per GPU and monitor until all finish
WORKER = Path(os.environ.get("TRACE_WORKER_PATH", "/kaggle/working/mammqa_trace_worker.py"))   # written by Cell 7
assert WORKER.exists(), "run Cell 7 first"
def launch(rank, world):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    out = open(DIRS["logs"] / f"worker{rank}.stdout", "a")
    return subprocess.Popen([sys.executable, str(WORKER), str(OUTPUT_ROOT), str(rank), str(world)], env=env, stdout=out, stderr=subprocess.STDOUT)

WORLD = max(1, N_GPU)
procs = {r: launch(r, WORLD) for r in range(WORLD)}
restarts = Counter()
last_report, t_start = 0.0, time.time()
done0 = len(list(DIRS["checkpoints"].glob("*.json")))
while procs:
    time.sleep(2 if MOCK else 60)
    for r, p in list(procs.items()):
        code = p.poll()
        if code is None:
            continue
        del procs[r]
        mine_left = [q for q in ORDER[r::WORLD] if not (DIRS["checkpoints"] / f"{q}.json").exists()]
        if code != 0 and mine_left and restarts[r] < 2 and time.time() < RUN_CONFIG["DEADLINE_EPOCH"]:
            restarts[r] += 1
            log(f"worker {r} exited with code {code}; restarting ({restarts[r]}/2)")
            procs[r] = launch(r, WORLD)
        else:
            log(f"worker {r} finished (exit {code}); {len(mine_left)} of its questions not done")
    done = len(list(DIRS["checkpoints"].glob("*.json")))
    if time.time() - last_report > (5 if MOCK else 600) or not procs:
        rate = (done - done0) / max(1e-9, (time.time() - t_start) / 3600)
        eta = (len(ORDER) - done) / rate if rate > 0 else float("nan")
        log(f"progress {done}/{len(ORDER)} | {rate:.1f} q/h | ETA {eta:.1f} h | errors {len(list(DIRS['errors'].glob('*.json')))}")
        last_report = time.time()

# %%
# Cell 9 — FREEZE predictions (hash) before any gold is read
done_qids = [q for q in ORDER if (DIRS["checkpoints"] / f"{q}.json").exists()]
PRED_PATH = DIRS["predictions"] / "mammqa_predictions.jsonl"
with open(PRED_PATH, "w", encoding="utf-8") as f:
    for q in done_qids:
        f.write(json.dumps(json.loads((DIRS["checkpoints"] / f"{q}.json").read_text(encoding="utf-8")), ensure_ascii=False, sort_keys=True) + "\n")
FROZEN = {"n": len(done_qids), "of": len(ORDER), "sha256": hashlib.sha256(PRED_PATH.read_bytes()).hexdigest(),
          "selection_sha256": SELECTION["sha256_questions"], "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
atomic_json(DIRS["predictions"] / "PREDICTIONS_FROZEN.json", FROZEN)
log(f"PREDICTIONS FROZEN: {FROZEN['n']}/{FROZEN['of']} sha256 {FROZEN['sha256'][:16]} — gold evaluation starts now")

# %%
# Cell 10 — official MMQA EM/F1 (DROP-style) after the freeze; the hop audit runs on mammqa_predictions.jsonl
def _is_num(t):
    try: float(t); return True
    except ValueError: return False
_P = set(string.punctuation)
def _nt(t):
    t = t.lower()
    if not _is_num(t): t = "".join(c for c in t if c not in _P)
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    if _is_num(t): t = str(float(t))
    return " ".join(t.split())
def _norm(s): return " ".join(x for x in (_nt(tok) for tok in re.split(" |-", str(s))) if x.strip()).strip()
def _f1(p, g):
    i = len(g & p); pr = 1.0 if not p else i / len(p); rc = 1.0 if not g else i / len(g)
    return 0.0 if pr == 0 and rc == 0 else 2 * pr * rc / (pr + rc)
def em_f1(pred, gold):
    from itertools import permutations
    pn = [_norm(x) for x in pred]; gn = [_norm(x) for x in gold]
    em = float(set(pn) == set(gn) and len(pn) == len(gn))
    pb = [set(x.split()) for x in pn]; gb = [set(x.split()) for x in gn]
    def ok(g, p):
        gnum = {w for w in g if _is_num(w)}
        return (not gnum) or bool(gnum & {w for w in p if _is_num(w)})
    S = [[_f1(p, g) if ok(g, p) else 0.0 for p in pb] for g in gb]
    try:
        from scipy.optimize import linear_sum_assignment; import numpy as np
        r, c = linear_sum_assignment(-np.array(S)); pairs = list(zip(r, c))
    except Exception:
        pairs = [(i, max(range(len(pb)), key=lambda j: S[i][j])) for i in range(len(gb))]
    best = [0.0] * max(len(gb), len(pb))
    for i, j in pairs: best[i] = max(best[i], S[i][j])
    return em, sum(best) / len(best)
def fmt(v): return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)

gold = {r["qid"]: r for r in read_jsonl(MMQA["dev"]) if r["qid"] in set(done_qids)}
rows = []
for line in open(PRED_PATH, encoding="utf-8"):
    p = json.loads(line); g = gold[p["id"]]
    golds = [fmt(a["answer"]) for a in g["answers"]]
    parts = [x.strip() for x in re.split(r"\s*\|\s*|\s*;\s*", p["prediction"]) if x.strip()] or [p["prediction"]]
    em, f1 = em_f1(parts, golds)
    rows.append({"qid": p["id"], "type": g["metadata"]["type"], "top_type": g["metadata"]["type"].split("(")[0], "em": em, "f1": round(f1, 4),
                 "prediction": p["prediction"], "gold": " | ".join(golds), "seconds": p["seconds"], "n_images": p["n_images"],
                 "hit_cap": sum(c["hit_max_new_tokens"] for c in p["calls"].values()),
                 "shortened": sum(c["evidence_shortenings"] for c in p["calls"].values())})
import csv
if rows:
    with open(DIRS["metrics"] / "per_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
by_type = defaultdict(list)
for r in rows: by_type[r["top_type"]].append(r["em"])
n_rows = max(1, len(rows))
SUMMARY = {"n": len(rows), "EM": round(sum(r["em"] for r in rows) / n_rows, 4), "F1": round(sum(r["f1"] for r in rows) / n_rows, 4),
           "EM_by_type": {t: [round(sum(v) / len(v), 4), len(v)] for t, v in sorted(by_type.items())},
           "mean_seconds_per_question": round(sum(r["seconds"] for r in rows) / n_rows, 1),
           "calls_hitting_token_cap": sum(r["hit_cap"] for r in rows), "evidence_shortenings": sum(r["shortened"] for r in rows),
           "errors": [p.stem for p in DIRS["errors"].glob("*.json") if not (DIRS["checkpoints"] / p.name).exists()],
           "frozen": FROZEN, "wall_hours": round((time.time() - NOTEBOOK_START) / 3600, 2)}
atomic_json(DIRS["metrics"] / "summary.json", SUMMARY)
print(json.dumps(SUMMARY, indent=1))
