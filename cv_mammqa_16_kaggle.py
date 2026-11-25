# %% [markdown]
# # Claim-Verified MAMMQA — Paired 16-Question Provided-Context Pilot
#
# This notebook implements a fair comparison between a **MAMMQA-style provided-context
# baseline** and **CV-MAMMQA** on the MultiModalQA development set.
#
# Experimental contract:
#
# - Provided official supporting context only; no retrieval, RAG, BM25, FAISS, embeddings,
#   graph search, web search, or external knowledge lookup.
# - The two methods use the same 16 questions, official evidence, Qwen2.5-VL backbone,
#   generation settings, and exactly shared Stage-I/Stage-II generations.
# - Gold answers, aliases, intermediate answers, and evaluation-only labels never enter a
#   model prompt. Predictions are persisted and frozen before evaluation begins.
# - The baseline sees only Stage-II outputs. Claim states, scores, verification notes, and
#   repairs are confined to the CV branch.
#
# Required Kaggle inputs: the MultiModalQA dataset (including official image pixels or its
# image ZIP), and either internet/model access to `Qwen/Qwen2.5-VL-7B-Instruct` or an
# attached local copy of that model. Select **2 × Tesla T4** as the accelerator.

# %%
# Cell 1 — environment diagnostics and only-if-missing dependencies.
# This cell never installs or changes torch, torchvision, torchaudio, CUDA, or NumPy.
import importlib.util
import platform
import subprocess
import sys

print("Python:", sys.version)
print("Platform:", platform.platform())
try:
    import torch
except Exception as exc:
    raise RuntimeError("Kaggle's existing PyTorch installation could not be imported.") from exc

print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available(), "GPU count:", torch.cuda.device_count())
for gpu_index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(gpu_index)
    print(gpu_index, props.name, f"{props.total_memory / 2**30:.2f} GiB")

required_packages = {
    "transformers": "transformers>=4.49,<6",
    "accelerate": "accelerate>=1,<2",
    "bitsandbytes": "bitsandbytes>=0.45,<1",
    "qwen_vl_utils": "qwen-vl-utils>=0.0.8,<1",
    "pandas": "pandas>=2,<3",
}
missing_specs = [spec for module, spec in required_packages.items() if importlib.util.find_spec(module) is None]
if missing_specs:
    print("Installing missing user-space packages only:", missing_specs)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_specs])
else:
    print("Required user-space packages are already available.")

# %%
# Cell 2 — imports, immutable experiment settings, output layout, and dataclasses.
import csv
import hashlib
import inspect
import json
import math
import os
import random
import re
import string
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

SELECTION_SEED = 20261013
GENERATION_SEED = 20261013
N_QUESTIONS = 16
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH_OVERRIDE = None  # Set to an attached local model folder only when auto-discovery is ambiguous.
DATA_ROOT_HINT = Path("/kaggle/input/datasets/adibraian/multimodalqa-dataset")
OUTPUT_ROOT = Path("/kaggle/working/cv_mammqa_16")
MAX_INPUT_TOKENS = 4096
STAGE1_MAX_NEW_TOKENS = 112
STAGE2_MAX_NEW_TOKENS = 128
AGGREGATOR_MAX_NEW_TOKENS = 80
CLAIM_MAX_NEW_TOKENS = 128
VERIFIER_MAX_NEW_TOKENS = 96
REPAIR_MAX_NEW_TOKENS = 112
TEMPERATURE = 0.3
TOP_P = 0.7
MAX_CLAIMS = 3
MAX_REPAIR_ROUNDS = 1
AGENT_SUPPORT_WEIGHT = 0.5
CLAIM_SCORE_MAP = {
    "SUPPORTED": 2,
    "PARTIAL": 1,
    "NOT_VERIFIABLE": 0,
    "UNSUPPORTED": -1,
    "CONTRADICTED": -2,
}
PROMPT_VERSIONS = {
    "stage1": "1.0",
    "stage2_anchor": "1.0",
    "baseline_aggregator": "1.0",
    "claim_extractor": "1.0",
    "claim_verifier": "1.0",
    "local_repair": "1.0",
    "cv_aggregator": "1.0",
    "format_repair": "1.0",
}
EXCLUDED_QIDS = set()  # Add prior DEBUG16_V1 / VALIDATION16_V2 / smoke qids here before first run.
RUN_PAIRED_BOOTSTRAP = True
BOOTSTRAP_RESAMPLES = 10_000

DIRS = {name: OUTPUT_ROOT / name for name in [
    "config", "selected_questions", "checkpoints", "predictions", "metrics", "diagnostics", "logs", "images"
]}
for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

random.seed(GENERATION_SEED)
np.random.seed(GENERATION_SEED)
torch.manual_seed(GENERATION_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GENERATION_SEED)

RUN_LOG = DIRS["logs"] / "run.log"


def log(message: str) -> None:
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(stamped + "\n")


@dataclass(frozen=True)
class QueryInput:
    qid: str
    question: str
    support_ids: Tuple[str, ...]
    support_modalities: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class EvaluationOnly:
    gold_answers: Tuple[str, ...]
    intermediate_answers: Any = None
    metadata_labels: Any = None


@dataclass(frozen=True)
class ProvidedContext:
    text: str
    table: str
    image_metadata: str
    image_paths: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    modalities: Tuple[str, ...]


@dataclass(frozen=True)
class InferenceInput:
    query: QueryInput
    context: ProvidedContext


@dataclass
class Stage1Output:
    agent: str
    answer: str
    insight: str
    raw_output: str


@dataclass
class Stage2Output:
    agent: str
    answer: str
    reason: str
    evidence_reference: str
    raw_output: str


@dataclass
class VerificationResult:
    status: str
    evidence_modality: str
    evidence_note: str
    raw_output: str
    parse_fallback: bool = False


@dataclass
class ClaimState:
    text: str
    verification: Optional[VerificationResult] = None


@dataclass
class CandidateState:
    answer: str
    normalized_answer: str
    supporting_agents: List[str]
    reasons: List[str]
    evidence_references: List[str]
    claims: List[ClaimState] = field(default_factory=list)
    claim_score: float = 0.0
    candidate_total_score: float = 0.0
    revision_occurred: bool = False
    original_answer: str = ""
    revised_answer: str = ""
    claim_extraction_fallback: bool = False


@dataclass
class RepairResult:
    triggered: bool
    agent: str = ""
    original_candidate: str = ""
    revised_candidate: str = ""
    revised_reason: str = ""
    changed: bool = False
    raw_output: str = ""
    parse_fallback: bool = False


@dataclass(frozen=True)
class PromptPacket:
    qid: str
    stage: str
    system: str
    protected_prefix: str
    evidence_blocks: Tuple[str, ...]
    protected_suffix: str
    image_paths: Tuple[str, ...] = ()


@dataclass
class MethodPrediction:
    answer: str
    raw_output: str
    seconds: float
    calls: int
    format_repair_used: bool = False


FORBIDDEN_PROMPT_KEYS = {
    "answers", "answer_aliases", "gold", "gold_answers", "intermediate_answers",
    "metadata_labels", "evaluationonly", "evaluation_only",
}


def assert_no_evaluation_fields(value: Any, where: str = "root") -> None:
    """Reject evaluation-field *keys*; answer strings may legitimately occur in supplied evidence."""
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_PROMPT_KEYS:
                raise AssertionError(f"Evaluation-only key leaked at {where}.{key}")
            assert_no_evaluation_fields(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_evaluation_fields(child, f"{where}[{index}]")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


log("Configuration initialized")

# %%
# Cell 3 — robust MMQA discovery, schema parsing, and hard inference/evaluation separation.
MMQA_FILENAMES = {
    "dev": "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_dev.jsonl",
    "texts": "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl",
    "tables": "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl",
    "images": "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl",
}
MMQA_SHORT_NAMES = {
    "dev": "MMQA_dev.jsonl",
    "texts": "MMQA_texts.jsonl",
    "tables": "MMQA_tables.jsonl",
    "images": "MMQA_images.jsonl",
}


def resolve_dataset_file(kind: str) -> Path:
    names = [MMQA_FILENAMES[kind], MMQA_SHORT_NAMES[kind]]
    search_roots = [DATA_ROOT_HINT, Path("/kaggle/input")]
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted({candidate.resolve() for name in names for candidate in root.rglob(name) if candidate.is_file()})
        if matches:
            preferred = [path for path in matches if MMQA_FILENAMES[kind] == path.name]
            pool = preferred or matches
            # Nested Kaggle exports occasionally expose the same file twice. Prefer the shortest path,
            # then require the chosen file to be non-empty and log every alternative.
            chosen = sorted(pool, key=lambda path: (len(path.parts), str(path)))[0]
            if chosen.stat().st_size == 0:
                raise RuntimeError(f"Resolved empty {kind} file: {chosen}")
            if len(pool) > 1:
                log(f"Multiple {kind} candidates found; deterministically chose {chosen}; alternatives={pool[1:]}")
            return chosen
    raise FileNotFoundError(
        f"Could not find {kind}. Attach the MultiModalQA Kaggle dataset; searched names {names} under {search_roots}."
    )


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def answer_strings(row: dict) -> Tuple[str, ...]:
    values: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            values.append(node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for key in ["answer", "answers", "aliases", "normalized_answer"]:
                if key in node:
                    walk(node[key])

    walk(row.get("answers", []))
    return tuple(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def extract_support_info(row: dict, known_ids: Dict[str, set]) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    ids: List[str] = []
    modalities: Dict[str, str] = {}

    def add(value: Any, modality: Optional[str] = None) -> None:
        identifier = str(value)
        if identifier in {"", "None"}:
            return
        inferred = modality or next((name for name, values in known_ids.items() if identifier in values), None)
        if inferred and identifier in known_ids[inferred]:
            ids.append(identifier)
            modalities[identifier] = inferred

    def walk(node: Any, modality: Optional[str] = None) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, modality)
        elif isinstance(node, dict):
            local = node.get("modality", node.get("doc_part", modality))
            local = {"text": "text", "table": "table", "image": "image"}.get(str(local).lower(), modality)
            for key, value in node.items():
                mapped = {
                    "text_doc_id": "text", "table_id": "table", "image_doc_id": "image", "doc_id": local
                }.get(key)
                if mapped:
                    add(value, mapped)
                elif key not in {"answer", "answers", "aliases", "text", "content", "intermediate_answers"}:
                    walk(value, local)
        elif modality:
            add(node, modality)

    walk(row.get("supporting_context", []))
    unique_ids = tuple(dict.fromkeys(ids))
    return unique_ids, tuple((identifier, modalities[identifier]) for identifier in unique_ids)


def load_mmqa() -> Tuple[Dict[str, QueryInput], Dict[str, Dict[str, dict]], Dict[str, Path]]:
    paths = {kind: resolve_dataset_file(kind) for kind in MMQA_FILENAMES}
    for kind, path in paths.items():
        log(f"{kind:7s}: {path}")
    dev_rows = load_jsonl(paths["dev"])
    text_rows = load_jsonl(paths["texts"])
    table_rows = load_jsonl(paths["tables"])
    image_rows = load_jsonl(paths["images"])
    corpora = {
        "text": {str(row["id"]): row for row in text_rows},
        "table": {str(row["id"]): row for row in table_rows},
        "image": {str(row["id"]): row for row in image_rows},
    }
    if any(len(corpora[name]) != len(rows) for name, rows in [("text", text_rows), ("table", table_rows), ("image", image_rows)]):
        raise RuntimeError("Duplicate corpus IDs detected.")
    known_ids = {name: set(records) for name, records in corpora.items()}
    inference: Dict[str, QueryInput] = {}
    for row in dev_rows:
        qid = str(row.get("qid", row.get("id", ""))).strip()
        question = str(row.get("question", "")).strip()
        if not qid or not question or qid in inference:
            raise RuntimeError(f"Malformed or duplicate dev record qid={qid!r}")
        support_ids, support_modalities = extract_support_info(row, known_ids)
        if not support_ids:
            raise RuntimeError(f"No official supporting context IDs parsed for qid={qid}")
        inference[qid] = QueryInput(qid, question, support_ids, support_modalities)
        # Deliberately do not read answers, intermediate answers, or metadata here.
        # The dev file is reopened for those fields only after predictions are frozen.
    del dev_rows, text_rows, table_rows, image_rows
    return inference, corpora, paths


INFERENCE_CATALOG, CORPORA, DATA_PATHS = load_mmqa()
print("Corpus sizes:", {name: len(records) for name, records in CORPORA.items()}, "dev:", len(INFERENCE_CATALOG))
assert all(set(asdict(query)) == {"qid", "question", "support_ids", "support_modalities"} for query in INFERENCE_CATALOG.values())
assert_no_evaluation_fields({qid: asdict(query) for qid, query in list(INFERENCE_CATALOG.items())[:3]}, "inference_catalog")

# %%
# Cell 4 — freeze a new deterministic, approximately stratified 16-question set.
TARGET_COMBINATIONS = [
    ("text", "image"),
    ("text", "table"),
    ("table", "image"),
    ("text", "table", "image"),
]
SELECTION_PATH = DIRS["selected_questions"] / "selected_questions.json"


def modality_combination(query: QueryInput) -> Tuple[str, ...]:
    return tuple(sorted({modality for _, modality in query.support_modalities}))


def select_question_ids() -> List[str]:
    eligible = [qid for qid in sorted(INFERENCE_CATALOG) if qid not in {str(value) for value in EXCLUDED_QIDS}]
    if len(eligible) < N_QUESTIONS:
        raise RuntimeError(f"Only {len(eligible)} eligible questions remain after exclusions.")
    rng = random.Random(SELECTION_SEED)
    strata: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for qid in eligible:
        strata[modality_combination(INFERENCE_CATALOG[qid])].append(qid)
    for values in strata.values():
        rng.shuffle(values)
    selected: List[str] = []
    quota = max(1, N_QUESTIONS // len(TARGET_COMBINATIONS))
    for combination in TARGET_COMBINATIONS:
        selected.extend(strata[tuple(sorted(combination))][:quota])
        strata[tuple(sorted(combination))] = strata[tuple(sorted(combination))][quota:]
    remaining = [qid for values in strata.values() for qid in values if qid not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: N_QUESTIONS - len(selected)])
    if len(selected) != N_QUESTIONS or len(set(selected)) != N_QUESTIONS:
        raise RuntimeError(f"Selection failed: obtained {len(selected)} unique={len(set(selected))}")
    return selected


if SELECTION_PATH.exists():
    saved_selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if not isinstance(saved_selection, list) or len(saved_selection) != N_QUESTIONS:
        raise RuntimeError("Existing selected_questions.json is malformed; move it aside to start a new frozen run.")
    SELECTED_QIDS = [str(item["qid"]) for item in saved_selection]
    if any(qid not in INFERENCE_CATALOG or qid in {str(value) for value in EXCLUDED_QIDS} for qid in SELECTED_QIDS):
        raise RuntimeError("Existing frozen selection conflicts with the current dataset or EXCLUDED_QIDS.")
    if any(int(item.get("selection_seed", -1)) != SELECTION_SEED for item in saved_selection):
        raise RuntimeError("Existing selection seed differs from SELECTION_SEED.")
    log("Reusing frozen selected_questions.json")
else:
    SELECTED_QIDS = select_question_ids()
    selection_payload = [
        {"qid": qid, "question": INFERENCE_CATALOG[qid].question, "selection_seed": SELECTION_SEED}
        for qid in SELECTED_QIDS
    ]
    assert_no_evaluation_fields(selection_payload, "selected_questions")
    atomic_json(SELECTION_PATH, selection_payload)
    log("Created new frozen selected_questions.json")

# Root-level alias plus the structured subdirectory copy requested by the experiment contract.
atomic_json(OUTPUT_ROOT / "selected_questions.json", json.loads(SELECTION_PATH.read_text(encoding="utf-8")))

print(pd.DataFrame([
    {"qid": qid, "modalities": "+".join(modality_combination(INFERENCE_CATALOG[qid])), "question": INFERENCE_CATALOG[qid].question}
    for qid in SELECTED_QIDS
]).to_string(index=False))
assert len(SELECTED_QIDS) == N_QUESTIONS and len(set(SELECTED_QIDS)) == N_QUESTIONS

# %%
# Cell 5 — official image discovery/validation and lossless provided-context construction.
def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def required_image_ids() -> List[str]:
    return sorted({
        identifier
        for qid in SELECTED_QIDS
        for identifier, modality in INFERENCE_CATALOG[qid].support_modalities
        if modality == "image"
    })


def prepare_image_manifest(image_ids: Sequence[str]) -> Dict[str, str]:
    if not image_ids:
        atomic_json(DIRS["diagnostics"] / "image_manifest.json", [])
        return {}
    expected_by_id = {}
    for image_id in image_ids:
        record = CORPORA["image"].get(image_id)
        if record is None:
            raise RuntimeError(f"Official image metadata missing for support ID {image_id}")
        expected_by_id[image_id] = Path(str(record.get("path", ""))).name
    if any(not filename for filename in expected_by_id.values()):
        raise RuntimeError("An official image record has no path filename.")

    input_root = Path("/kaggle/input")
    roots = []
    if input_root.exists():
        roots.extend(path for path in input_root.rglob("final_dataset_images") if path.is_dir())
        roots.extend(path for path in input_root.rglob("images") if path.is_dir() and "multimodal" in str(path).lower())
    roots = list(dict.fromkeys(path.resolve() for path in roots))

    filenames = set(expected_by_id.values())
    by_name: Dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.name in filenames:
                if path.name in by_name and by_name[path.name] != path.resolve():
                    raise RuntimeError(f"Ambiguous official image filename {path.name}: {by_name[path.name]} and {path}")
                by_name[path.name] = path.resolve()

    missing_names = filenames - set(by_name)
    if missing_names and input_root.exists():
        zip_candidates = []
        for path in input_root.rglob("*.zip"):
            if re.search(r"(mmqa|multimodal).*image|image.*(mmqa|multimodal)|final_dataset_images", path.name, re.I):
                try:
                    with zipfile.ZipFile(path) as archive:
                        archived_names = {Path(name).name for name in archive.namelist()}
                    if missing_names & archived_names:
                        zip_candidates.append(path)
                except zipfile.BadZipFile:
                    continue
        if len(zip_candidates) > 1:
            raise RuntimeError(f"Multiple official image ZIPs match required files: {zip_candidates}")
        if zip_candidates:
            extracted = DIRS["images"] / "official"
            marker = extracted / ".complete"
            if not marker.exists():
                safe_extract_zip(zip_candidates[0], extracted)
                marker.touch()
            roots.append(extracted)
            for path in extracted.rglob("*"):
                if path.is_file() and path.name in missing_names:
                    by_name[path.name] = path.resolve()

    # Last-resort local scan is still official-file discovery, never network retrieval.
    missing_names = filenames - set(by_name)
    if missing_names and input_root.exists():
        for path in input_root.rglob("*"):
            if path.is_file() and path.name in missing_names:
                if path.name in by_name and by_name[path.name] != path.resolve():
                    raise RuntimeError(f"Ambiguous official image filename {path.name}")
                by_name[path.name] = path.resolve()

    manifest = []
    resolved: Dict[str, str] = {}
    for image_id, filename in expected_by_id.items():
        path = by_name.get(filename)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"Official image bytes missing for id={image_id}, filename={filename}. "
                "Attach the official MultiModalQA image package/ZIP; replacement or web images are forbidden."
            )
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
        except Exception as exc:
            raise RuntimeError(f"PIL could not validate official image {path}: {exc}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        resolved[image_id] = str(path)
        manifest.append({
            "image_id": image_id, "metadata_path": CORPORA["image"][image_id].get("path"),
            "resolved_path": str(path), "width": width, "height": height,
            "format": image_format, "sha256": digest,
        })
    atomic_json(DIRS["diagnostics"] / "image_manifest.json", manifest)
    log(f"Validated {len(manifest)} required official image files")
    return resolved


def cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("text", cell.get("value", cell.get("column_name", ""))))
    return str(cell)


def table_to_readable(record: dict) -> str:
    table = record.get("table", record)
    header_raw = table.get("header", [])
    rows_raw = table.get("table_rows", table.get("rows", []))
    headers = [cell_text(value).replace("\n", " ").strip() for value in header_raw]
    rows = [[cell_text(value).replace("\n", " ").strip() for value in row] for row in rows_raw]
    width = max([len(headers), *(len(row) for row in rows)] or [0])
    headers = (headers + [f"column_{index + 1}" for index in range(len(headers), width)])[:width]
    rendered = [f"[TABLE: {record.get('title', '')}]", "COLUMNS: " + " | ".join(headers)]
    for row_index, row in enumerate(rows, 1):
        padded = row + [""] * (width - len(row))
        rendered.append(f"row {row_index}: " + " ; ".join(f"{key} = {value}" for key, value in zip(headers, padded)))
    return "\n".join(rendered)


IMAGE_PATHS = prepare_image_manifest(required_image_ids())


def build_provided_context(query: QueryInput) -> ProvidedContext:
    text_blocks: List[str] = []
    table_blocks: List[str] = []
    image_blocks: List[str] = []
    image_paths: List[str] = []
    modality_lookup = dict(query.support_modalities)
    for identifier in query.support_ids:
        modality = modality_lookup[identifier]
        record = CORPORA[modality].get(identifier)
        if record is None:
            raise RuntimeError(f"Provided support ID {identifier} ({modality}) missing from corpus")
        if modality == "text":
            text_blocks.append(f"[TEXT: {record.get('title', '')} | id={identifier}]\n{record.get('text', record.get('paragraph', ''))}")
        elif modality == "table":
            table_blocks.append(table_to_readable(record) + f"\nTABLE_ID: {identifier}")
        elif modality == "image":
            path = IMAGE_PATHS.get(identifier)
            if not path:
                raise RuntimeError(f"Required official image path unresolved for {identifier}")
            image_blocks.append(f"[IMAGE: {record.get('title', '')} | id={identifier} | official_file={Path(path).name}]")
            image_paths.append(path)
    context = ProvidedContext(
        text="\n\n".join(text_blocks),
        table="\n\n".join(table_blocks),
        image_metadata="\n".join(image_blocks),
        image_paths=tuple(image_paths),
        source_ids=query.support_ids,
        modalities=tuple(sorted(set(modality_lookup.values()))),
    )
    assert set(context.source_ids) == set(query.support_ids)
    return context


INFERENCE_INPUTS = {
    qid: InferenceInput(INFERENCE_CATALOG[qid], build_provided_context(INFERENCE_CATALOG[qid]))
    for qid in SELECTED_QIDS
}
for item in INFERENCE_INPUTS.values():
    assert_no_evaluation_fields(asdict(item), f"inference_input[{item.query.qid}]")
print("LEAKAGE CHECK PASSED")

# %%
# Cell 6 — implementation fingerprint, restart isolation, and runtime configuration.
FINGERPRINT_INPUT = {
    "implementation": "CV-MAMMQA-provided-context-v1.0",
    "model": MODEL_ID,
    "selection_seed": SELECTION_SEED,
    "generation_seed": GENERATION_SEED,
    "selected_qids": SELECTED_QIDS,
    "prompt_versions": PROMPT_VERSIONS,
    "generation": {"do_sample": True, "temperature": TEMPERATURE, "top_p": TOP_P},
    "deterministic_utility_generation": {"do_sample": False},
    "token_limits": {
        "input": MAX_INPUT_TOKENS, "stage1": STAGE1_MAX_NEW_TOKENS,
        "stage2": STAGE2_MAX_NEW_TOKENS, "aggregator": AGGREGATOR_MAX_NEW_TOKENS,
        "claim": CLAIM_MAX_NEW_TOKENS, "verifier": VERIFIER_MAX_NEW_TOKENS,
        "repair": REPAIR_MAX_NEW_TOKENS,
    },
    "claim_score_map": CLAIM_SCORE_MAP,
    "agent_support_weight": AGENT_SUPPORT_WEIGHT,
    "max_claims": MAX_CLAIMS,
    "max_repair_rounds": MAX_REPAIR_ROUNDS,
    "repair_rule": "one highest-priority problematic candidate branch; never rerun Stage I/II",
}
IMPLEMENTATION_FINGERPRINT = stable_hash(FINGERPRINT_INPUT)
CHECKPOINT_ROOT = DIRS["checkpoints"] / IMPLEMENTATION_FINGERPRINT
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_CONFIG = {
    **FINGERPRINT_INPUT,
    "implementation_fingerprint": IMPLEMENTATION_FINGERPRINT,
    "quantization": {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float16",
        "device_map": "balanced",
    },
    "experiment_setting": "PROVIDED_CONTEXT",
    "retrieval": "NONE",
    "data_paths": {key: str(path) for key, path in DATA_PATHS.items()},
}
atomic_json(DIRS["config"] / "runtime_config.json", RUNTIME_CONFIG)
log(f"Implementation fingerprint: {IMPLEMENTATION_FINGERPRINT}")


def context_digest(item: InferenceInput) -> str:
    image_hashes = []
    for path_string in item.context.image_paths:
        path = Path(path_string)
        image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return stable_hash({"input": asdict(item), "image_sha256": image_hashes})


CONTEXT_DIGESTS = {qid: context_digest(item) for qid, item in INFERENCE_INPUTS.items()}

# %%
# Cell 7 — one 4-bit, two-GPU Qwen load and protected multimodal generation wrapper.
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL = None
PROCESSOR = None
GENERATION_DIAGNOSTICS: List[dict] = []
TRUNCATION_DIAGNOSTICS: List[dict] = []
CALL_COUNTER = Counter()


def resolve_model_source() -> str:
    if MODEL_PATH_OVERRIDE:
        override = Path(str(MODEL_PATH_OVERRIDE))
        if not (override / "config.json").is_file():
            raise FileNotFoundError(f"MODEL_PATH_OVERRIDE is not a model directory: {override}")
        return str(override)
    candidates = []
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for config_path in input_root.rglob("config.json"):
            path_lower = str(config_path.parent).lower()
            if "qwen" not in path_lower or "vl" not in path_lower:
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(config.get("model_type", "")).lower() in {"qwen2_5_vl", "qwen2-vl", "qwen2_vl"}:
                candidates.append(config_path.parent.resolve())
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))
    if len(candidates) > 1:
        exact = [path for path in candidates if "7b" in str(path).lower() and "instruct" in str(path).lower()]
        if len(exact) == 1:
            return str(exact[0])
        raise RuntimeError(f"Multiple local Qwen-VL models found: {candidates}. Set MODEL_PATH_OVERRIDE explicitly.")
    return str(candidates[0]) if candidates else MODEL_ID


def load_qwen_once() -> None:
    global MODEL, PROCESSOR
    if MODEL is not None or PROCESSOR is not None:
        if MODEL is None or PROCESSOR is None:
            raise RuntimeError("Partial model initialization detected.")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This notebook requires Kaggle 2 × T4 GPU for the specified balanced 4-bit run.")
    source = resolve_model_source()
    log(f"Loading Qwen once from: {source}")
    PROCESSOR = AutoProcessor.from_pretrained(source, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source,
        quantization_config=quantization,
        device_map="balanced",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    device_map = getattr(MODEL, "hf_device_map", {})
    atomic_json(DIRS["diagnostics"] / "model_device_map.json", device_map)
    log(f"Model loaded; device-map entries={len(device_map)}")


def stable_call_seed(qid: str, stage: str) -> int:
    digest = hashlib.sha256(f"{GENERATION_SEED}|{qid}|{stage}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def shorten_token_text(text: str, budget: int) -> str:
    if not text or budget <= 0:
        return ""
    token_ids = PROCESSOR.tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= budget:
        return text
    marker = "\n[... EVIDENCE SHORTENED DETERMINISTICALLY ...]\n"
    marker_ids = PROCESSOR.tokenizer.encode(marker, add_special_tokens=False)
    available = max(2, budget - len(marker_ids))
    head_count = max(1, int(available * 0.7))
    tail_count = max(1, available - head_count)
    return (
        PROCESSOR.tokenizer.decode(token_ids[:head_count], skip_special_tokens=True)
        + marker
        + PROCESSOR.tokenizer.decode(token_ids[-tail_count:], skip_special_tokens=True)
    )


def packet_messages(packet: PromptPacket, evidence_blocks: Sequence[str]) -> List[dict]:
    assert_no_evaluation_fields(asdict(packet), f"prompt_packet[{packet.stage}]")
    content = []
    for image_path in packet.image_paths:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Prompt image missing: {path}")
        content.append({"type": "image", "image": f"file://{path}"})
    full_text = packet.protected_prefix + "\n\n" + "\n\n".join(block for block in evidence_blocks if block) + "\n\n" + packet.protected_suffix
    content.append({"type": "text", "text": full_text})
    return [{"role": "system", "content": packet.system}, {"role": "user", "content": content}]


def processor_batch(messages: List[dict]):
    prompt = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    kwargs = {"text": [prompt], "return_tensors": "pt", "padding": True}
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs
    return PROCESSOR(**kwargs)


def fit_packet(packet: PromptPacket):
    full_blocks = list(packet.evidence_blocks)
    batch = processor_batch(packet_messages(packet, full_blocks))
    full_tokens = int(batch.input_ids.shape[1])
    if full_tokens <= MAX_INPUT_TOKENS:
        return batch, full_blocks, False, full_tokens

    empty_batch = processor_batch(packet_messages(packet, ["" for _ in full_blocks]))
    protected_tokens = int(empty_batch.input_ids.shape[1])
    if protected_tokens > MAX_INPUT_TOKENS:
        raise RuntimeError(
            f"Protected prompt + image tokens exceed MAX_INPUT_TOKENS for {packet.qid}/{packet.stage}: {protected_tokens}"
        )
    original_token_counts = [len(PROCESSOR.tokenizer.encode(block, add_special_tokens=False)) for block in full_blocks]
    low, high = 0, sum(original_token_counts)
    best_batch, best_blocks = empty_batch, ["" for _ in full_blocks]
    while low <= high:
        total_budget = (low + high) // 2
        nonempty = [index for index, count in enumerate(original_token_counts) if count]
        allocations = [0] * len(full_blocks)
        if nonempty:
            base = total_budget // len(nonempty)
            remainder = total_budget % len(nonempty)
            for rank, index in enumerate(nonempty):
                allocations[index] = min(original_token_counts[index], base + (1 if rank < remainder else 0))
            unused = total_budget - sum(allocations)
            while unused > 0:
                progressed = False
                for index in nonempty:
                    room = original_token_counts[index] - allocations[index]
                    if room > 0 and unused > 0:
                        take = min(room, unused)
                        allocations[index] += take
                        unused -= take
                        progressed = True
                if not progressed:
                    break
        candidate_blocks = [shorten_token_text(block, allocation) for block, allocation in zip(full_blocks, allocations)]
        candidate_batch = processor_batch(packet_messages(packet, candidate_blocks))
        candidate_tokens = int(candidate_batch.input_ids.shape[1])
        if candidate_tokens <= MAX_INPUT_TOKENS:
            best_batch, best_blocks = candidate_batch, candidate_blocks
            low = total_budget + 1
        else:
            high = total_budget - 1
    final_tokens = int(best_batch.input_ids.shape[1])
    event = {
        "qid": packet.qid, "stage": packet.stage, "original_input_tokens": full_tokens,
        "final_input_tokens": final_tokens, "protected_tokens": protected_tokens,
        "evidence_shortened": True,
    }
    TRUNCATION_DIAGNOSTICS.append(event)
    append_jsonl(DIRS["diagnostics"] / "evidence_shortening.jsonl", event)
    return best_batch, best_blocks, True, full_tokens


def generate_packet(packet: PromptPacket, max_new_tokens: int, deterministic: bool) -> str:
    load_qwen_once()
    started = time.perf_counter()
    batch, fitted_blocks, evidence_shortened, original_tokens = fit_packet(packet)
    input_tokens = int(batch.input_ids.shape[1])
    target_device = getattr(MODEL, "device", torch.device("cuda:0"))
    batch = batch.to(target_device)
    seed = stable_call_seed(packet.qid, packet.stage)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not deterministic,
    }
    if not deterministic:
        generation_kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
    with torch.inference_mode():
        generated = MODEL.generate(**batch, **generation_kwargs)
    new_ids = generated[:, batch.input_ids.shape[1]:]
    output_tokens = int(new_ids.shape[1])
    raw = PROCESSOR.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    seconds = time.perf_counter() - started
    record = {
        "qid": packet.qid, "stage": packet.stage, "deterministic": deterministic,
        "input_tokens": input_tokens, "original_input_tokens": original_tokens,
        "output_tokens": output_tokens, "max_new_tokens": max_new_tokens,
        "hit_max_new_tokens": output_tokens >= max_new_tokens,
        "evidence_shortened": evidence_shortened, "seconds": seconds,
        "image_count": len(packet.image_paths), "seed": seed,
    }
    GENERATION_DIAGNOSTICS.append(record)
    CALL_COUNTER[packet.stage] += 1
    append_jsonl(DIRS["diagnostics"] / "generation_calls.jsonl", record)
    return raw

# %%
# Cell 8 — parsers, modality prompts, and exactly shared Stage I/Stage II.
INSUFFICIENT_VALUES = {"", "insufficient", "abstain", "unknown", "n/a", "none", "not enough information"}


def parse_labeled(raw: str, labels: Sequence[str]) -> Dict[str, str]:
    result = {label: "" for label in labels}
    label_pattern = "|".join(re.escape(label) for label in labels)
    matches = list(re.finditer(rf"(?im)^\s*({label_pattern})\s*:\s*", raw))
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        result[label] = raw[match.end():end].strip()
    return result


def clean_candidate(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value.strip(" \t\n\r\"'`*.,;:")


def normalize_candidate(value: str) -> str:
    value = clean_candidate(value).lower()
    value = value.strip(string.punctuation + " ")
    value = re.sub(r"\s+", " ", value)
    words = value.split()
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join(words)


def is_concrete(value: str) -> bool:
    return normalize_candidate(value) not in INSUFFICIENT_VALUES


def modality_evidence(context: ProvidedContext, modality: str) -> Tuple[str, Tuple[str, ...]]:
    if modality == "text":
        return context.text or "[NO TEXT EVIDENCE]", ()
    if modality == "table":
        return context.table or "[NO TABLE EVIDENCE]", ()
    if modality == "image":
        return context.image_metadata or "[NO IMAGE EVIDENCE]", context.image_paths
    raise ValueError(modality)


def run_stage1_modality(item: InferenceInput, modality: str) -> Stage1Output:
    if modality not in item.context.modalities:
        return Stage1Output(modality, "INSUFFICIENT", "This modality is not present in the supplied context.", "NOT_RUN")
    evidence, image_paths = modality_evidence(item.context, modality)
    role = f"You are the {modality} expert."
    packet = PromptPacket(
        qid=item.query.qid,
        stage=f"stage1_{modality}",
        system="You are a careful multimodal question-answering expert. Use only the evidence explicitly supplied.",
        protected_prefix=f"{role}\n\nQuestion:\n{item.query.question}",
        evidence_blocks=(f"{modality.upper()} evidence:\n{evidence}",),
        protected_suffix=(
            f"Use only this {modality} modality. Identify information relevant to answering the question.\n"
            "Return exactly:\nANSWER: <candidate or INSUFFICIENT>\nINSIGHT: <maximum 2 concise sentences>\n"
            "Do not use information from other modalities."
        ),
        image_paths=image_paths,
    )
    raw = generate_packet(packet, STAGE1_MAX_NEW_TOKENS, deterministic=False)
    parsed = parse_labeled(raw, ["ANSWER", "INSIGHT"])
    answer = clean_candidate(parsed["ANSWER"]) or "INSUFFICIENT"
    insight = parsed["INSIGHT"].strip() or raw.strip()[:600]
    return Stage1Output(modality, answer, insight, raw)


def raw_other_evidence(context: ProvidedContext, anchor: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    blocks: List[str] = []
    images: Tuple[str, ...] = ()
    if anchor != "text" and "text" in context.modalities:
        blocks.append("OTHER MODALITY — TEXT:\n" + context.text)
    if anchor != "table" and "table" in context.modalities:
        blocks.append("OTHER MODALITY — TABLE:\n" + context.table)
    if anchor != "image" and "image" in context.modalities:
        blocks.append("OTHER MODALITY — IMAGE METADATA (pixels are attached):\n" + context.image_metadata)
        images = context.image_paths
    if not blocks:
        blocks.append("[NO OTHER MODALITY IS AVAILABLE]")
    return tuple(blocks), images


def run_stage2_anchor(item: InferenceInput, anchor: str, stage1: Stage1Output) -> Stage2Output:
    other_blocks, other_images = raw_other_evidence(item.context, anchor)
    evidence_blocks = (
        f"Trusted {anchor.upper()} anchor insight:\nANSWER: {stage1.answer}\nINSIGHT: {stage1.insight}",
        *other_blocks,
    )
    packet = PromptPacket(
        qid=item.query.qid,
        stage=f"stage2_{anchor}_anchor",
        system="You are a concise cross-modal synthesis agent. Use only supplied evidence.",
        protected_prefix=f"You are the {anchor.upper()} cross-modal synthesis agent.\n\nQuestion:\n{item.query.question}",
        evidence_blocks=tuple(evidence_blocks),
        protected_suffix=(
            "Use the anchor insight as one reasoning perspective and use the raw other-modality evidence to solve the question.\n"
            "Return exactly:\nANSWER: <concise answer or INSUFFICIENT>\n"
            "REASON: <maximum 2 concise sentences>\nEVIDENCE: <short evidence reference>\n"
            "Do not produce long explanations."
        ),
        image_paths=other_images,
    )
    raw = generate_packet(packet, STAGE2_MAX_NEW_TOKENS, deterministic=False)
    parsed = parse_labeled(raw, ["ANSWER", "REASON", "EVIDENCE"])
    answer = clean_candidate(parsed["ANSWER"]) or "INSUFFICIENT"
    reason = parsed["REASON"].strip() or raw.strip()[:600]
    evidence_reference = parsed["EVIDENCE"].strip()
    return Stage2Output(f"{anchor}_anchor", answer, reason, evidence_reference, raw)


def run_shared_stages(item: InferenceInput) -> Tuple[Dict[str, Stage1Output], Dict[str, Stage2Output], float]:
    started = time.perf_counter()
    stage1 = {
        modality: run_stage1_modality(item, modality)
        for modality in ["text", "table", "image"]
        if modality in item.context.modalities
    }
    stage2 = {
        modality: run_stage2_anchor(item, modality, stage1[modality])
        for modality in ["text", "table", "image"]
        if modality in stage1
    }
    if not stage2:
        raise RuntimeError(f"No Stage-II agent was available for {item.query.qid}")
    return stage1, stage2, time.perf_counter() - started

# %%
# Cell 9 — uncontaminated MAMMQA baseline aggregation and deterministic candidate memory.
def parse_final(raw: str) -> str:
    parsed = parse_labeled(raw, ["FINAL"])["FINAL"]
    return clean_candidate(parsed)


def format_repair_final(qid: str, method: str, raw: str) -> str:
    packet = PromptPacket(
        qid=qid,
        stage=f"{method}_format_repair",
        system="You repair output formatting only. Do not change the proposed answer.",
        protected_prefix="Rewrite the existing output into the required one-line schema.",
        evidence_blocks=("EXISTING OUTPUT:\n" + raw,),
        protected_suffix="Return ONLY:\nFINAL: <the same concise answer or ABSTAIN>",
    )
    repaired = generate_packet(packet, 48, deterministic=True)
    return parse_final(repaired)


def aggregate_mammqa_baseline(item: InferenceInput, stage2: Dict[str, Stage2Output]) -> MethodPrediction:
    started = time.perf_counter()
    calls_before = sum(CALL_COUNTER.values())
    blocks = tuple(
        f"{output.agent.upper()} SYNTHESIS:\nANSWER: {output.answer}\nREASON: {output.reason}\nEVIDENCE: {output.evidence_reference}"
        for output in stage2.values()
    )
    packet = PromptPacket(
        qid=item.query.qid,
        stage="baseline_aggregator",
        system="You are the MAMMQA-style consensus aggregator. You receive synthesis outputs, not raw gold labels.",
        protected_prefix=f"Question:\n{item.query.question}\n\nSynthesis agents:",
        evidence_blocks=blocks,
        protected_suffix=(
            "Resolve disagreement hierarchically:\n"
            "1. Prefer coherent agreement among multiple agents.\n"
            "2. A strongly justified candidate may beat uncertain agents.\n"
            "3. When concrete answers disagree, compare reasoning quality.\n"
            "4. Return a concise final answer.\n"
            "You have no claim-verification results, scores, dispute labels, or repair outputs.\n"
            "Return ONLY:\nFINAL: <answer>"
        ),
    )
    raw = generate_packet(packet, AGGREGATOR_MAX_NEW_TOKENS, deterministic=False)
    answer = parse_final(raw)
    repair_used = False
    if not answer:
        answer = format_repair_final(item.query.qid, "baseline", raw)
        repair_used = True
    if not answer:
        answer = clean_candidate(raw.splitlines()[0] if raw.splitlines() else "") or "ABSTAIN"
    return MethodPrediction(
        answer=answer,
        raw_output=raw,
        seconds=time.perf_counter() - started,
        calls=sum(CALL_COUNTER.values()) - calls_before,
        format_repair_used=repair_used,
    )


def agreement_analysis(stage2: Dict[str, Stage2Output]) -> dict:
    concrete = [output for output in stage2.values() if is_concrete(output.answer)]
    counts = Counter(normalize_candidate(output.answer) for output in concrete)
    if counts:
        consensus_normalized, consensus_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        consensus_answer = next(output.answer for output in concrete if normalize_candidate(output.answer) == consensus_normalized)
    else:
        consensus_answer, consensus_count = "", 0
    unique_count = len(counts)
    if not concrete:
        agreement_type = "no_candidate"
    elif unique_count == 1 and len(concrete) == len(stage2):
        agreement_type = "unanimous"
    elif consensus_count >= 2:
        agreement_type = "majority_agreement" if unique_count > 1 else "single_candidate_with_nonvotes"
    elif unique_count == 1:
        agreement_type = "single_concrete_candidate"
    else:
        agreement_type = "full_disagreement"
    return {
        "concrete_candidate_count": len(concrete),
        "unique_candidate_count": unique_count,
        "consensus_answer": consensus_answer,
        "consensus_count": consensus_count,
        "stage2_disagreement": unique_count >= 2,
        "agreement_type": agreement_type,
    }


def build_candidate_memory(stage2: Dict[str, Stage2Output]) -> List[CandidateState]:
    grouped: Dict[str, List[Stage2Output]] = defaultdict(list)
    for output in stage2.values():
        if is_concrete(output.answer):
            grouped[normalize_candidate(output.answer)].append(output)
    memory = []
    for normalized in sorted(grouped):
        outputs = grouped[normalized]
        memory.append(CandidateState(
            answer=outputs[0].answer,
            normalized_answer=normalized,
            supporting_agents=[output.agent for output in outputs],
            reasons=[output.reason for output in outputs if output.reason],
            evidence_references=[output.evidence_reference for output in outputs if output.evidence_reference],
            original_answer=outputs[0].answer,
        ))
    return memory

# %%
# Cell 10 — CV-MAMMQA claim extraction, evidence verification, one selective repair, and verified aggregation.
def claim_token_set(text: str) -> set:
    return set(normalize_candidate(text).split())


def claims_near_duplicate(left: str, right: str, threshold: float = 0.8) -> bool:
    left_tokens, right_tokens = claim_token_set(left), claim_token_set(right)
    if not left_tokens or not right_tokens:
        return normalize_candidate(left) == normalize_candidate(right)
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens)) >= threshold


def parse_claims(raw: str, fallback_reason: str) -> Tuple[List[str], bool]:
    claims = []
    for match in re.finditer(r"(?im)^\s*CLAIM\s+[1-3]\s*:\s*(.+?)(?=\n\s*CLAIM\s+[1-3]\s*:|\Z)", raw, re.S):
        claim = re.sub(r"\s+", " ", match.group(1)).strip(" -*\n")
        if claim and not any(claims_near_duplicate(claim, existing) for existing in claims):
            claims.append(claim)
    if claims:
        return claims[:MAX_CLAIMS], False
    fallback = re.sub(r"\s+", " ", fallback_reason).strip()
    return ([fallback] if fallback else ["The candidate is the answer to the question."]), True


def extract_claims(item: InferenceInput, candidate: CandidateState) -> CandidateState:
    reasoning = "\n".join(f"- {reason}" for reason in candidate.reasons) or "[NO PARSED REASON]"
    packet = PromptPacket(
        qid=item.query.qid,
        stage=f"claim_extraction_{candidate.normalized_answer[:32]}",
        system="You are a deterministic claim extractor. Do not introduce facts.",
        protected_prefix=(
            f"You are a claim extractor.\n\nQuestion:\n{item.query.question}\n\n"
            f"Candidate answer:\n{candidate.answer}"
        ),
        evidence_blocks=("Agent reasoning:\n" + reasoning,),
        protected_suffix=(
            "Convert the reasoning into the MINIMUM set of atomic factual claims needed to justify the answer.\n"
            "Each claim must be independently verifiable, concise, factual, and relevant. Do NOT add new facts.\n"
            "Return exactly:\nCLAIM 1: ...\nCLAIM 2: ...\nCLAIM 3: ...\nMaximum 3 claims."
        ),
    )
    raw = generate_packet(packet, CLAIM_MAX_NEW_TOKENS, deterministic=True)
    extracted, fallback = parse_claims(raw, " ".join(candidate.reasons))
    deduplicated = []
    for claim in extracted:
        if not any(claims_near_duplicate(claim, existing) for existing in deduplicated):
            deduplicated.append(claim)
    candidate.claims = [ClaimState(text=claim) for claim in deduplicated[:MAX_CLAIMS]]
    candidate.claim_extraction_fallback = fallback
    return candidate


def verification_evidence_blocks(context: ProvidedContext) -> Tuple[str, ...]:
    return (
        "Provided text:\n" + (context.text or "[NONE]"),
        "Provided table:\n" + (context.table or "[NONE]"),
        "Provided image metadata (official image pixels are attached when available):\n" + (context.image_metadata or "[NONE]"),
    )


def parse_verification(raw: str) -> VerificationResult:
    parsed = parse_labeled(raw, ["STATUS", "EVIDENCE_MODALITY", "EVIDENCE_NOTE"])
    status_match = re.search(r"\b(SUPPORTED|PARTIAL|UNSUPPORTED|CONTRADICTED|NOT_VERIFIABLE)\b", parsed["STATUS"].upper())
    modality_match = re.search(r"\b(text|table|image|multiple|none)\b", parsed["EVIDENCE_MODALITY"].lower())
    if not status_match:
        return VerificationResult("NOT_VERIFIABLE", "none", "Verifier output could not be parsed.", raw, True)
    return VerificationResult(
        status=status_match.group(1),
        evidence_modality=modality_match.group(1) if modality_match else "none",
        evidence_note=re.sub(r"\s+", " ", parsed["EVIDENCE_NOTE"]).strip() or "No parseable evidence note.",
        raw_output=raw,
        parse_fallback=False,
    )


def verify_claim(item: InferenceInput, candidate_answer: str, claim: str, stage_suffix: str = "") -> VerificationResult:
    packet = PromptPacket(
        qid=item.query.qid,
        stage=f"claim_verification{stage_suffix}",
        system="You are a strict deterministic multimodal evidence verifier. Use no outside knowledge.",
        protected_prefix=(
            "You have been given the complete context available for this question.\n\n"
            f"Question:\n{item.query.question}\n\nCandidate answer:\n{candidate_answer}\n\nClaim:\n{claim}"
        ),
        evidence_blocks=verification_evidence_blocks(item.context),
        protected_suffix=(
            "Judge ONLY whether the supplied context supports this specific claim. Agent disagreement is not evidence.\n"
            "Labels:\nSUPPORTED: directly supported.\nPARTIAL: some elements supported but an essential part is missing.\n"
            "UNSUPPORTED: context does not establish it.\nCONTRADICTED: context is inconsistent with it.\n"
            "NOT_VERIFIABLE: cannot be checked from supplied context.\n"
            "Return ONLY:\nSTATUS: <label>\nEVIDENCE_MODALITY: <text|table|image|multiple|none>\n"
            "EVIDENCE_NOTE: <one concise sentence>"
        ),
        image_paths=item.context.image_paths,
    )
    raw = generate_packet(packet, VERIFIER_MAX_NEW_TOKENS, deterministic=True)
    return parse_verification(raw)


def score_candidate(candidate: CandidateState) -> CandidateState:
    statuses = [claim.verification.status if claim.verification else "NOT_VERIFIABLE" for claim in candidate.claims]
    candidate.claim_score = float(sum(CLAIM_SCORE_MAP[status] for status in statuses))
    candidate.candidate_total_score = candidate.claim_score + AGENT_SUPPORT_WEIGHT * (len(candidate.supporting_agents) - 1)
    return candidate


def verify_candidate(item: InferenceInput, candidate: CandidateState, stage_suffix: str = "") -> CandidateState:
    for index, claim in enumerate(candidate.claims):
        if claim.verification is None:
            claim.verification = verify_claim(
                item, candidate.answer, claim.text, stage_suffix=f"{stage_suffix}_{index + 1}"
            )
    return score_candidate(candidate)


def status_counts(memory: Sequence[CandidateState]) -> Counter:
    return Counter(
        claim.verification.status if claim.verification else "NOT_VERIFIABLE"
        for candidate in memory for claim in candidate.claims
    )


def detect_dispute(memory: Sequence[CandidateState], agreement: dict) -> bool:
    if agreement["unique_candidate_count"] >= 2:
        return True
    if not memory:
        return False
    only_problem = any(
        claim.verification and claim.verification.status in {"PARTIAL", "UNSUPPORTED", "CONTRADICTED"}
        for claim in memory[0].claims
    )
    consensus_norm = normalize_candidate(agreement.get("consensus_answer", ""))
    majority = next((candidate for candidate in memory if candidate.normalized_answer == consensus_norm), None)
    majority_contradicted = bool(majority and any(
        claim.verification and claim.verification.status == "CONTRADICTED" for claim in majority.claims
    ))
    return only_problem or majority_contradicted


def select_repair_branch(memory: Sequence[CandidateState], agreement: dict) -> Optional[CandidateState]:
    problematic = []
    severity = {"CONTRADICTED": 3, "UNSUPPORTED": 2, "PARTIAL": 1}
    consensus_norm = normalize_candidate(agreement.get("consensus_answer", ""))
    for candidate in memory:
        worst = max((severity.get(claim.verification.status, 0) for claim in candidate.claims if claim.verification), default=0)
        if worst:
            problematic.append((
                candidate.normalized_answer != consensus_norm,
                -worst,
                -len(candidate.supporting_agents),
                -candidate.candidate_total_score,
                candidate.normalized_answer,
                candidate,
            ))
    return sorted(problematic, key=lambda item: item[:-1])[0][-1] if problematic else None


def parse_repair(raw: str, candidate: CandidateState) -> RepairResult:
    parsed = parse_labeled(raw, ["REVISED_ANSWER", "REVISED_REASON", "CHANGED"])
    revised = clean_candidate(parsed["REVISED_ANSWER"])
    reason = parsed["REVISED_REASON"].strip()
    changed_token = parsed["CHANGED"].strip().lower()
    if not revised or changed_token not in {"yes", "no"}:
        return RepairResult(
            triggered=True,
            agent=candidate.supporting_agents[0] if candidate.supporting_agents else "",
            original_candidate=candidate.answer,
            revised_candidate=candidate.answer,
            revised_reason=" ".join(candidate.reasons),
            changed=False,
            raw_output=raw,
            parse_fallback=True,
        )
    changed = changed_token == "yes" and (
        normalize_candidate(revised) != candidate.normalized_answer
        or normalize_candidate(reason) != normalize_candidate(" ".join(candidate.reasons))
    )
    return RepairResult(
        triggered=True,
        agent=candidate.supporting_agents[0] if candidate.supporting_agents else "",
        original_candidate=candidate.answer,
        revised_candidate=revised if changed else candidate.answer,
        revised_reason=reason if changed and reason else " ".join(candidate.reasons),
        changed=changed,
        raw_output=raw,
        parse_fallback=False,
    )


def repair_candidate(item: InferenceInput, candidate: CandidateState) -> RepairResult:
    problems = [
        claim for claim in candidate.claims
        if claim.verification and claim.verification.status in {"PARTIAL", "UNSUPPORTED", "CONTRADICTED"}
    ]
    if not problems:
        return RepairResult(triggered=False)
    problem_text = "\n".join(
        f"- CLAIM: {claim.text}\n  STATUS: {claim.verification.status}\n  EVIDENCE NOTE: {claim.verification.evidence_note}"
        for claim in problems
    )
    packet = PromptPacket(
        qid=item.query.qid,
        stage="local_repair",
        system="You revise exactly one disputed reasoning branch using only supplied evidence.",
        protected_prefix=(
            f"Question:\n{item.query.question}\n\nOriginal candidate:\n{candidate.answer}\n\n"
            f"Original reasoning:\n{' '.join(candidate.reasons)}\n\nProblematic claim(s) and verifier judgment:\n{problem_text}"
        ),
        evidence_blocks=verification_evidence_blocks(item.context),
        protected_suffix=(
            "Reconsider ONLY the disputed reasoning. Preserve already-supported claims. Do NOT restart the entire solution.\n"
            "Return exactly:\nREVISED_ANSWER: <answer or INSUFFICIENT>\n"
            "REVISED_REASON: <maximum 2 sentences>\nCHANGED: <yes|no>"
        ),
        image_paths=item.context.image_paths,
    )
    raw = generate_packet(packet, REPAIR_MAX_NEW_TOKENS, deterministic=False)
    return parse_repair(raw, candidate)


def apply_and_reverify_repair(item: InferenceInput, candidate: CandidateState, repair: RepairResult) -> CandidateState:
    if not repair.changed:
        return candidate
    old_supported = {
        normalize_candidate(claim.text): claim.verification
        for claim in candidate.claims
        if claim.verification and claim.verification.status == "SUPPORTED"
    }
    candidate.revision_occurred = True
    candidate.revised_answer = repair.revised_candidate
    candidate.answer = repair.revised_candidate
    candidate.normalized_answer = normalize_candidate(repair.revised_candidate)
    candidate.reasons = [repair.revised_reason]
    candidate.claims = []
    candidate = extract_claims(item, candidate)
    for claim in candidate.claims:
        previous = old_supported.get(normalize_candidate(claim.text))
        if previous is not None:
            claim.verification = previous
    return verify_candidate(item, candidate, stage_suffix="_revised")


def verified_memory_text(memory: Sequence[CandidateState]) -> str:
    if not memory:
        return "[NO CONCRETE CANDIDATE]"
    blocks = []
    for index, candidate in enumerate(memory, 1):
        claim_lines = []
        for claim in candidate.claims:
            verification = claim.verification or VerificationResult("NOT_VERIFIABLE", "none", "Missing verification", "", True)
            claim_lines.append(
                f"- {claim.text}\n  STATUS={verification.status}; MODALITY={verification.evidence_modality}; NOTE={verification.evidence_note}"
            )
        blocks.append(
            f"CANDIDATE {index}: {candidate.answer}\n"
            f"SUPPORTING_AGENTS: {', '.join(candidate.supporting_agents)}\n"
            f"CLAIM_SCORE: {candidate.claim_score}\nTOTAL_SCORE: {candidate.candidate_total_score}\n"
            f"REVISION_OCCURRED: {candidate.revision_occurred}\nCLAIMS:\n" + "\n".join(claim_lines)
        )
    return "\n\n".join(blocks)


def aggregate_cv_mammqa(
    item: InferenceInput,
    memory: Sequence[CandidateState],
    stage2: Dict[str, Stage2Output],
    started: float,
    calls_before: int,
) -> MethodPrediction:
    stage2_text = "\n\n".join(
        f"{output.agent}: ANSWER={output.answer}; REASON={output.reason}; EVIDENCE={output.evidence_reference}"
        for output in stage2.values()
    )
    packet = PromptPacket(
        qid=item.query.qid,
        stage="cv_aggregator",
        system="You are the final multimodal answer aggregator. Use verified claims, not simple majority voting.",
        protected_prefix=f"Question:\n{item.query.question}\n\nVerified candidate memory:",
        evidence_blocks=(verified_memory_text(memory), "Original Stage-II outputs:\n" + stage2_text),
        protected_suffix=(
            "Choose the answer best supported by VERIFIED claims.\n"
            "1. Prefer directly supported essential claims.\n2. Reject contradicted essential claims.\n"
            "3. Agent agreement is weaker than direct verification.\n4. INSUFFICIENT agents are neutral.\n"
            "5. If no candidate has enough support, return ABSTAIN. Do not abstain merely because some claims are uncertain.\n"
            "Return ONLY:\nFINAL: <concise answer or ABSTAIN>"
        ),
    )
    raw = generate_packet(packet, AGGREGATOR_MAX_NEW_TOKENS, deterministic=False)
    answer = parse_final(raw)
    repair_used = False
    if not answer:
        answer = format_repair_final(item.query.qid, "cv", raw)
        repair_used = True
    if not answer:
        answer = "ABSTAIN"
    return MethodPrediction(
        answer=answer,
        raw_output=raw,
        seconds=time.perf_counter() - started,
        calls=sum(CALL_COUNTER.values()) - calls_before,
        format_repair_used=repair_used,
    )


def run_cv_branch(item: InferenceInput, stage2: Dict[str, Stage2Output]) -> Tuple[MethodPrediction, dict]:
    started = time.perf_counter()
    calls_before = sum(CALL_COUNTER.values())
    agreement = agreement_analysis(stage2)
    memory = build_candidate_memory(stage2)
    for candidate in memory:
        extract_claims(item, candidate)
        verify_candidate(item, candidate)
    dispute = detect_dispute(memory, agreement)
    repair = RepairResult(triggered=False)
    if dispute and MAX_REPAIR_ROUNDS:
        branch = select_repair_branch(memory, agreement)
        if branch is not None:
            repair = repair_candidate(item, branch)
            apply_and_reverify_repair(item, branch, repair)
    prediction = aggregate_cv_mammqa(item, memory, stage2, started, calls_before)
    counts = status_counts(memory)
    diagnostics = {
        "agreement": agreement,
        "dispute_detected": dispute,
        "candidate_memory": [asdict(candidate) for candidate in memory],
        "repair": asdict(repair),
        "claim_counts": {status: counts.get(status, 0) for status in CLAIM_SCORE_MAP},
    }
    return prediction, diagnostics

# %%
# Cell 11 — paired execution with per-question atomic checkpoints and safe restart.
def checkpoint_path(qid: str) -> Path:
    return CHECKPOINT_ROOT / f"{qid}.json"


def validate_checkpoint(payload: dict, qid: str) -> None:
    required = {"fingerprint", "qid", "question", "context_digest", "shared", "baseline", "cv", "cv_diagnostics"}
    if not required <= set(payload):
        raise RuntimeError(f"Incomplete checkpoint for {qid}: missing {required - set(payload)}")
    if payload["fingerprint"] != IMPLEMENTATION_FINGERPRINT:
        raise RuntimeError(f"Fingerprint mismatch in checkpoint for {qid}")
    if payload["question"] != INFERENCE_INPUTS[qid].query.question:
        raise RuntimeError(f"Question mismatch in checkpoint for {qid}")
    if payload["context_digest"] != CONTEXT_DIGESTS[qid]:
        raise RuntimeError(f"Provided-context digest mismatch in checkpoint for {qid}")
    assert_no_evaluation_fields(payload, f"checkpoint[{qid}]")


def run_one_question(qid: str) -> dict:
    path = checkpoint_path(qid)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_checkpoint(payload, qid)
        log(f"RESUME {qid}: valid completed checkpoint")
        return payload
    item = INFERENCE_INPUTS[qid]
    diagnostics_start = len(GENERATION_DIAGNOSTICS)
    log(f"START {qid}: {item.query.question}")
    stage1, stage2, shared_seconds = run_shared_stages(item)
    # Freeze the exact shared objects before branching. Both branches consume the same stage2 instance.
    shared_digest_before = stable_hash({"stage1": {key: asdict(value) for key, value in stage1.items()},
                                        "stage2": {key: asdict(value) for key, value in stage2.items()}})
    baseline = aggregate_mammqa_baseline(item, stage2)
    cv, cv_diagnostics = run_cv_branch(item, stage2)
    shared_digest_after = stable_hash({"stage1": {key: asdict(value) for key, value in stage1.items()},
                                       "stage2": {key: asdict(value) for key, value in stage2.items()}})
    if shared_digest_before != shared_digest_after:
        raise RuntimeError(f"A branch mutated shared Stage-I/II outputs for {qid}")
    payload = {
        "fingerprint": IMPLEMENTATION_FINGERPRINT,
        "qid": qid,
        "question": item.query.question,
        "context_digest": CONTEXT_DIGESTS[qid],
        "source_ids": list(item.context.source_ids),
        "modalities": list(item.context.modalities),
        "shared": {
            "stage1": {key: asdict(value) for key, value in stage1.items()},
            "stage2": {key: asdict(value) for key, value in stage2.items()},
            "shared_seconds": shared_seconds,
            "shared_digest": shared_digest_before,
        },
        "baseline": asdict(baseline),
        "cv": asdict(cv),
        "cv_diagnostics": cv_diagnostics,
        "generation_diagnostics": GENERATION_DIAGNOSTICS[diagnostics_start:],
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    }
    assert_no_evaluation_fields(payload, f"checkpoint[{qid}]")
    atomic_json(path, payload)
    log(f"DONE {qid}: baseline={baseline.answer!r}; cv={cv.answer!r}")
    return payload


QUESTION_RESULTS: Dict[str, dict] = {}
for position, qid in enumerate(SELECTED_QIDS, 1):
    log(f"Progress {position}/{N_QUESTIONS}")
    QUESTION_RESULTS[qid] = run_one_question(qid)

assert set(QUESTION_RESULTS) == set(SELECTED_QIDS)
for qid, payload in QUESTION_RESULTS.items():
    validate_checkpoint(payload, qid)
log("All paired inference checkpoints complete")

# %%
# Cell 12 — write gold-free prediction files, hash them, and freeze inference.
def write_jsonl_atomic(path: Path, rows: Sequence[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            assert_no_evaluation_fields(row, f"prediction_file[{path.name}]")
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


shared_rows = []
baseline_rows = []
cv_rows = []
for qid in SELECTED_QIDS:
    result = QUESTION_RESULTS[qid]
    shared_rows.append({
        "fingerprint": IMPLEMENTATION_FINGERPRINT, "qid": qid, "question": result["question"],
        "context_digest": result["context_digest"], **result["shared"],
    })
    baseline_rows.append({
        "fingerprint": IMPLEMENTATION_FINGERPRINT, "qid": qid, "question": result["question"],
        "prediction": result["baseline"]["answer"], "raw_output": result["baseline"]["raw_output"],
        "seconds": result["baseline"]["seconds"], "calls": result["baseline"]["calls"],
        "format_repair_used": result["baseline"]["format_repair_used"],
    })
    cv_rows.append({
        "fingerprint": IMPLEMENTATION_FINGERPRINT, "qid": qid, "question": result["question"],
        "prediction": result["cv"]["answer"], "raw_output": result["cv"]["raw_output"],
        "seconds": result["cv"]["seconds"], "calls": result["cv"]["calls"],
        "format_repair_used": result["cv"]["format_repair_used"],
        "diagnostics": result["cv_diagnostics"],
    })

PREDICTION_PATHS = {
    "shared": DIRS["predictions"] / "predictions_shared.jsonl",
    "baseline": DIRS["predictions"] / "predictions_mammqa.jsonl",
    "cv": DIRS["predictions"] / "predictions_cv_mammqa.jsonl",
}
write_jsonl_atomic(PREDICTION_PATHS["shared"], shared_rows)
write_jsonl_atomic(PREDICTION_PATHS["baseline"], baseline_rows)
write_jsonl_atomic(PREDICTION_PATHS["cv"], cv_rows)
prediction_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in PREDICTION_PATHS.items()}
FREEZE_MARKER = DIRS["predictions"] / "PREDICTIONS_FROZEN.json"
atomic_json(FREEZE_MARKER, {
    "predictions_frozen_before_gold": True,
    "fingerprint": IMPLEMENTATION_FINGERPRINT,
    "qids": SELECTED_QIDS,
    "sha256": prediction_hashes,
    "frozen_utc": pd.Timestamp.utcnow().isoformat(),
})
PREDICTIONS_FROZEN = True
print("PREDICTIONS FROZEN.")
print("GOLD EVALUATION STARTS NOW.")

# %%
# Cell 13 — gold evaluation begins only after the freeze marker and hash checks.
def verify_prediction_freeze() -> None:
    if not PREDICTIONS_FROZEN or not FREEZE_MARKER.is_file():
        raise RuntimeError("Predictions are not frozen; gold access is forbidden.")
    marker = json.loads(FREEZE_MARKER.read_text(encoding="utf-8"))
    if marker.get("fingerprint") != IMPLEMENTATION_FINGERPRINT or marker.get("qids") != SELECTED_QIDS:
        raise RuntimeError("Freeze marker does not match this experiment.")
    for name, path in PREDICTION_PATHS.items():
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != marker["sha256"][name]:
            raise RuntimeError(f"Frozen prediction file changed: {path}")


ARTICLES = {"a", "an", "the"}


def normalize_answer(value: str) -> str:
    value = str(value).lower()
    value = "".join(" " if character in string.punctuation else character for character in value)
    return " ".join(word for word in value.split() if word not in ARTICLES)


def exact_match(prediction: str, golds: Sequence[str]) -> int:
    return int(any(normalize_answer(prediction) == normalize_answer(gold) for gold in golds))


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    scores = []
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        common = Counter(prediction_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if not prediction_tokens and not gold_tokens:
            scores.append(1.0)
        elif not prediction_tokens or not gold_tokens or overlap == 0:
            scores.append(0.0)
        else:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(gold_tokens)
            scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def is_abstention(answer: str) -> bool:
    return normalize_candidate(answer) in {"abstain", "insufficient", "not enough information", ""}


def stage2_answers(result: dict) -> Dict[str, str]:
    return {modality: output["answer"] for modality, output in result["shared"]["stage2"].items()}


def comparison_labels(baseline_correct: bool, cv_correct: bool, baseline_abstain: bool, cv_abstain: bool) -> Tuple[str, str]:
    if baseline_correct and cv_correct:
        return "BOTH_CORRECT", "BOTH_CORRECT"
    if baseline_correct:
        specific = "CV_ABSTAIN_BASELINE_CORRECT" if cv_abstain else "CV_CHANGED_CORRECT_TO_WRONG"
        return specific, "BASELINE_ONLY_CORRECT"
    if cv_correct:
        specific = "BASELINE_ABSTAIN_CV_CORRECT" if baseline_abstain else "CV_CHANGED_WRONG_TO_CORRECT"
        return specific, "CV_ONLY_CORRECT"
    return "BOTH_WRONG", "BOTH_WRONG"


def candidate_matches_gold(answer: str, golds: Sequence[str]) -> bool:
    return bool(exact_match(answer, golds))


def load_evaluation_after_freeze(dev_path: Path, selected_qids: Sequence[str]) -> Dict[str, EvaluationOnly]:
    verify_prediction_freeze()
    selected_set = set(selected_qids)
    evaluation: Dict[str, EvaluationOnly] = {}
    for row in load_jsonl(dev_path):
        qid = str(row.get("qid", row.get("id", ""))).strip()
        if qid not in selected_set:
            continue
        evaluation[qid] = EvaluationOnly(
            gold_answers=answer_strings(row),
            intermediate_answers=row.get("intermediate_answers"),
            metadata_labels=row.get("metadata"),
        )
    if set(evaluation) != selected_set:
        raise RuntimeError(f"Evaluation reload mismatch; missing={selected_set - set(evaluation)}")
    return evaluation


verify_prediction_freeze()
EVALUATION_VIEW = load_evaluation_after_freeze(DATA_PATHS["dev"], SELECTED_QIDS)  # First answer/metadata access.
PER_QUESTION_ROWS: List[dict] = []
FAILURE_ROWS: List[dict] = []

for qid in SELECTED_QIDS:
    result = QUESTION_RESULTS[qid]
    evaluation = EVALUATION_VIEW[qid]
    golds = list(evaluation.gold_answers)
    if not golds:
        raise RuntimeError(f"No accepted gold answers parsed for {qid}")
    baseline_prediction = result["baseline"]["answer"]
    cv_prediction = result["cv"]["answer"]
    baseline_em = exact_match(baseline_prediction, golds)
    cv_em = exact_match(cv_prediction, golds)
    baseline_f1 = token_f1(baseline_prediction, golds)
    cv_f1 = token_f1(cv_prediction, golds)
    baseline_abstain = is_abstention(baseline_prediction)
    cv_abstain = is_abstention(cv_prediction)
    category, generic_category = comparison_labels(bool(baseline_em), bool(cv_em), baseline_abstain, cv_abstain)
    s2_answers = stage2_answers(result)
    s2_gold = any(candidate_matches_gold(answer, golds) for answer in s2_answers.values())
    diagnostics = result["cv_diagnostics"]
    memory = diagnostics["candidate_memory"]
    claim_counts = diagnostics["claim_counts"]
    repair = diagnostics["repair"]
    repaired_before_correct = bool(repair["triggered"] and candidate_matches_gold(repair["original_candidate"], golds))
    repaired_after_correct = bool(repair["triggered"] and candidate_matches_gold(repair["revised_candidate"], golds))
    correct_memory = [candidate for candidate in memory if candidate_matches_gold(candidate["answer"], golds)]
    max_score = max((candidate["candidate_total_score"] for candidate in memory), default=-math.inf)
    verifier_favored_gold = bool(correct_memory and any(candidate["candidate_total_score"] == max_score for candidate in correct_memory))
    stage_diagnostics = result.get("generation_diagnostics", [])
    shared_calls = sum(record["stage"].startswith("stage1_") or record["stage"].startswith("stage2_") for record in stage_diagnostics)
    baseline_calls = sum(record["stage"].startswith("baseline_") for record in stage_diagnostics)
    cv_calls = len(stage_diagnostics) - shared_calls - baseline_calls
    row = {
        "qid": qid,
        "question": result["question"],
        "gold": " | ".join(golds),
        "baseline_prediction": baseline_prediction,
        "cv_prediction": cv_prediction,
        "baseline_em": baseline_em,
        "cv_em": cv_em,
        "baseline_f1": baseline_f1,
        "cv_f1": cv_f1,
        "stage2_text_answer": s2_answers.get("text", "INSUFFICIENT"),
        "stage2_table_answer": s2_answers.get("table", "INSUFFICIENT"),
        "stage2_image_answer": s2_answers.get("image", "INSUFFICIENT"),
        "stage2_unique_candidate_count": diagnostics["agreement"]["unique_candidate_count"],
        "stage2_consensus_answer": diagnostics["agreement"]["consensus_answer"],
        "stage2_consensus_count": diagnostics["agreement"]["consensus_count"],
        "stage2_disagreement": diagnostics["agreement"]["stage2_disagreement"],
        "stage2_agreement_type": diagnostics["agreement"]["agreement_type"],
        "stage2_consensus_correct": candidate_matches_gold(diagnostics["agreement"]["consensus_answer"], golds),
        "gold_present_in_stage2": s2_gold,
        "claims_extracted": sum(claim_counts.values()),
        "supported_claims": claim_counts.get("SUPPORTED", 0),
        "partial_claims": claim_counts.get("PARTIAL", 0),
        "unsupported_claims": claim_counts.get("UNSUPPORTED", 0),
        "contradicted_claims": claim_counts.get("CONTRADICTED", 0),
        "not_verifiable_claims": claim_counts.get("NOT_VERIFIABLE", 0),
        "dispute_detected": diagnostics["dispute_detected"],
        "repair_triggered": repair["triggered"],
        "repair_agent": repair["agent"],
        "original_candidate": repair["original_candidate"],
        "revised_candidate": repair["revised_candidate"],
        "repair_changed": repair["changed"],
        "repair_candidate_before_correct": repaired_before_correct,
        "repair_candidate_after_correct": repaired_after_correct,
        "baseline_correct": bool(baseline_em),
        "cv_correct": bool(cv_em),
        "baseline_abstain": baseline_abstain,
        "cv_abstain": cv_abstain,
        "comparison_category": category,
        "generic_comparison_category": generic_category,
        "shared_seconds": result["shared"]["shared_seconds"],
        "baseline_specific_seconds": result["baseline"]["seconds"],
        "cv_specific_seconds": result["cv"]["seconds"],
        "shared_calls": shared_calls,
        "baseline_specific_calls": baseline_calls,
        "cv_additional_calls": cv_calls,
        "verifier_favored_gold_candidate": verifier_favored_gold,
        "claim_rescue": bool(not baseline_em and cv_em and s2_gold and verifier_favored_gold),
        "claim_damage": bool(baseline_em and not cv_em),
    }
    PER_QUESTION_ROWS.append(row)

    if not cv_em:
        if not s2_gold:
            failure = "NO_CORRECT_STAGE2_CANDIDATE"
        elif cv_abstain or not candidate_matches_gold(cv_prediction, golds):
            if repair["triggered"] and repaired_before_correct and not repaired_after_correct:
                failure = "REPAIR_HARMED"
            elif repair["triggered"] and not repaired_after_correct:
                failure = "REPAIR_FAILED"
            elif correct_memory and any(
                claim["verification"] and claim["verification"]["status"] in {"UNSUPPORTED", "CONTRADICTED"}
                for candidate in correct_memory for claim in candidate["claims"]
            ):
                failure = "VERIFIER_FALSE_NEGATIVE"
            elif correct_memory and not verifier_favored_gold:
                failure = "VERIFIER_FALSE_POSITIVE"
            elif any(candidate.get("claim_extraction_fallback") for candidate in correct_memory):
                failure = "CLAIM_EXTRACTION_FAILURE"
            elif correct_memory:
                failure = "FINAL_AGGREGATION_FAILURE"
            else:
                failure = "CORRECT_CANDIDATE_LOST"
        else:
            failure = "ANSWER_FORMAT_FAILURE"
        FAILURE_ROWS.append({"qid": qid, "failure_type": failure, "cv_prediction": cv_prediction, "gold": " | ".join(golds)})

PER_QUESTION_DF = pd.DataFrame(PER_QUESTION_ROWS)
PER_QUESTION_DF.to_csv(DIRS["metrics"] / "per_question_comparison.csv", index=False)
pd.DataFrame(FAILURE_ROWS, columns=["qid", "failure_type", "cv_prediction", "gold"]).to_csv(
    DIRS["metrics"] / "failure_analysis.csv", index=False
)
print(PER_QUESTION_DF[["qid", "baseline_prediction", "cv_prediction", "baseline_em", "cv_em", "comparison_category"]].to_string(index=False))

# %%
# Cell 14 — aggregate metrics, bootstrap, mechanism diagnostics, and required CSV exports.
def safe_mean(series: Iterable[float]) -> float:
    values = [float(value) for value in series]
    return float(np.mean(values)) if values else math.nan


summary_rows = []
for method, prefix in [("MAMMQA_BASELINE", "baseline"), ("CV_MAMMQA", "cv")]:
    correct = int(PER_QUESTION_DF[f"{prefix}_em"].sum())
    if prefix == "baseline":
        total_calls = PER_QUESTION_DF["shared_calls"] + PER_QUESTION_DF["baseline_specific_calls"]
        total_runtime = PER_QUESTION_DF["shared_seconds"] + PER_QUESTION_DF["baseline_specific_seconds"]
    else:
        total_calls = PER_QUESTION_DF["shared_calls"] + PER_QUESTION_DF["cv_additional_calls"]
        total_runtime = PER_QUESTION_DF["shared_seconds"] + PER_QUESTION_DF["cv_specific_seconds"]
    summary_rows.append({
        "method": method,
        "n": len(PER_QUESTION_DF),
        "exact_correct": correct,
        "EM": safe_mean(PER_QUESTION_DF[f"{prefix}_em"]),
        "mean_token_F1": safe_mean(PER_QUESTION_DF[f"{prefix}_f1"]),
        "incorrect_count": len(PER_QUESTION_DF) - correct,
        "abstention_count": int(PER_QUESTION_DF[f"{prefix}_abstain"].sum()),
        "average_total_calls": safe_mean(total_calls),
        "average_runtime": safe_mean(total_runtime),
    })
SUMMARY_DF = pd.DataFrame(summary_rows)
SUMMARY_DF.to_csv(DIRS["metrics"] / "summary.csv", index=False)

disagreement = PER_QUESTION_DF[PER_QUESTION_DF["stage2_disagreement"]]
agreement = PER_QUESTION_DF[
    PER_QUESTION_DF["stage2_agreement_type"].isin(["unanimous", "majority_agreement"])
]
disagreement_with_gold = disagreement[disagreement["gold_present_in_stage2"]]
cv_help = int(((PER_QUESTION_DF["baseline_em"] == 0) & (PER_QUESTION_DF["cv_em"] == 1)).sum())
cv_harm = int(((PER_QUESTION_DF["baseline_em"] == 1) & (PER_QUESTION_DF["cv_em"] == 0)).sum())
disagreement_row = {
    "total_disagreement_questions": len(disagreement),
    "gold_candidate_available_in_disagreement": int(disagreement["gold_present_in_stage2"].sum()),
    "baseline_correct_disagreement": int(disagreement["baseline_em"].sum()),
    "cv_correct_disagreement": int(disagreement["cv_em"].sum()),
    "baseline_EM_disagreement": safe_mean(disagreement["baseline_em"]),
    "cv_EM_disagreement": safe_mean(disagreement["cv_em"]),
    "baseline_F1_disagreement": safe_mean(disagreement["baseline_f1"]),
    "cv_F1_disagreement": safe_mean(disagreement["cv_f1"]),
    "disagreement_resolution_accuracy_baseline": safe_mean(disagreement_with_gold["baseline_em"]),
    "disagreement_resolution_accuracy_cv": safe_mean(disagreement_with_gold["cv_em"]),
    "agreement_questions": len(agreement),
    "baseline_EM_agreement": safe_mean(agreement["baseline_em"]),
    "cv_EM_agreement": safe_mean(agreement["cv_em"]),
    "baseline_F1_agreement": safe_mean(agreement["baseline_f1"]),
    "cv_F1_agreement": safe_mean(agreement["cv_f1"]),
    "CV_help_count": cv_help,
    "CV_harm_count": cv_harm,
    "net_gain": cv_help - cv_harm,
}
pd.DataFrame([disagreement_row]).to_csv(DIRS["metrics"] / "disagreement_analysis.csv", index=False)

repair_subset = PER_QUESTION_DF[PER_QUESTION_DF["repair_triggered"]]
repair_success_count = int(((repair_subset["baseline_em"] == 0) & (repair_subset["cv_em"] == 1)).sum())
repair_candidate_rescue = int((
    (~repair_subset["repair_candidate_before_correct"]) & repair_subset["repair_candidate_after_correct"]
).sum())
repair_harm_count = int(((repair_subset["baseline_em"] == 1) & (repair_subset["cv_em"] == 0)).sum())
claim_summary = {
    "total_claims": int(PER_QUESTION_DF["claims_extracted"].sum()),
    "supported": int(PER_QUESTION_DF["supported_claims"].sum()),
    "partial": int(PER_QUESTION_DF["partial_claims"].sum()),
    "unsupported": int(PER_QUESTION_DF["unsupported_claims"].sum()),
    "contradicted": int(PER_QUESTION_DF["contradicted_claims"].sum()),
    "not_verifiable": int(PER_QUESTION_DF["not_verifiable_claims"].sum()),
    "average_claims_per_question": safe_mean(PER_QUESTION_DF["claims_extracted"]),
    "average_verification_calls_per_question": safe_mean([
        sum(record["stage"].startswith("claim_verification") for record in QUESTION_RESULTS[qid].get("generation_diagnostics", []))
        for qid in SELECTED_QIDS
    ]),
    "questions_with_claim_verification": int((PER_QUESTION_DF["claims_extracted"] > 0).sum()),
    "questions_with_dispute": int(PER_QUESTION_DF["dispute_detected"].sum()),
    "questions_with_repair": len(repair_subset),
    "repair_activation_rate": len(repair_subset) / len(PER_QUESTION_DF),
    "repair_success_count": repair_success_count,
    "repair_candidate_rescue_count": repair_candidate_rescue,
    "repair_harm_count": repair_harm_count,
    "claim_rescue_count": int(PER_QUESTION_DF["claim_rescue"].sum()),
    "claim_damage_count": int(PER_QUESTION_DF["claim_damage"].sum()),
}
pd.DataFrame([claim_summary]).to_csv(DIRS["metrics"] / "claim_verification_summary.csv", index=False)

gold_stage2_subset = PER_QUESTION_DF[PER_QUESTION_DF["gold_present_in_stage2"]]
candidate_survival = {
    "questions_with_gold_in_stage2": len(gold_stage2_subset),
    "baseline_candidate_survival": safe_mean(gold_stage2_subset["baseline_em"]),
    "cv_candidate_survival": safe_mean(gold_stage2_subset["cv_em"]),
}
atomic_json(DIRS["metrics"] / "candidate_survival.json", candidate_survival)

runtime_summary = {
    "shared_stage1_stage2_total_seconds": float(PER_QUESTION_DF["shared_seconds"].sum()),
    "baseline_specific_total_seconds": float(PER_QUESTION_DF["baseline_specific_seconds"].sum()),
    "cv_specific_total_seconds": float(PER_QUESTION_DF["cv_specific_seconds"].sum()),
    "average_baseline_equivalent_calls_per_question": safe_mean(
        PER_QUESTION_DF["shared_calls"] + PER_QUESTION_DF["baseline_specific_calls"]
    ),
    "average_additional_cv_calls_per_question": safe_mean(PER_QUESTION_DF["cv_additional_calls"]),
    "paired_run_total_calls": int((
        PER_QUESTION_DF["shared_calls"] + PER_QUESTION_DF["baseline_specific_calls"] + PER_QUESTION_DF["cv_additional_calls"]
    ).sum()),
    "logical_cost_note": "Shared calls count once experimentally but belong to both logical methods.",
}
atomic_json(DIRS["metrics"] / "runtime_summary.json", runtime_summary)

all_call_records = [record for result in QUESTION_RESULTS.values() for record in result.get("generation_diagnostics", [])]
TOKEN_DIAGNOSTICS_DF = pd.DataFrame(all_call_records)
TOKEN_DIAGNOSTICS_DF.to_csv(DIRS["diagnostics"] / "token_cap_diagnostics.csv", index=False)
token_summary = {
    "calls": len(TOKEN_DIAGNOSTICS_DF),
    "hit_max_new_tokens": int(TOKEN_DIAGNOSTICS_DF["hit_max_new_tokens"].sum()) if len(TOKEN_DIAGNOSTICS_DF) else 0,
    "evidence_shortenings": int(TOKEN_DIAGNOSTICS_DF["evidence_shortened"].sum()) if len(TOKEN_DIAGNOSTICS_DF) else 0,
    "protected_truncations": 0,
    "oom_count": 0,
}
atomic_json(DIRS["diagnostics"] / "token_cap_summary.json", token_summary)

bootstrap_result = {"enabled": RUN_PAIRED_BOOTSTRAP, "resamples": 0}
if RUN_PAIRED_BOOTSTRAP:
    differences = (PER_QUESTION_DF["cv_f1"] - PER_QUESTION_DF["baseline_f1"]).to_numpy(dtype=float)
    rng = np.random.default_rng(SELECTION_SEED)
    resampled = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        resampled[index] = differences[rng.integers(0, len(differences), len(differences))].mean()
    bootstrap_result = {
        "enabled": True,
        "resamples": BOOTSTRAP_RESAMPLES,
        "mean_delta_f1": float(differences.mean()),
        "ci95_percentile": [float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))],
        "label": "Exploratory paired bootstrap; n=16, not proof of significance.",
        "paired_em_wins_cv": cv_help,
        "paired_em_wins_baseline": cv_harm,
        "paired_em_ties": len(PER_QUESTION_DF) - cv_help - cv_harm,
    }
atomic_json(DIRS["metrics"] / "paired_bootstrap.json", bootstrap_result)

research_metrics = {
    "delta_EM": float(SUMMARY_DF.loc[SUMMARY_DF.method == "CV_MAMMQA", "EM"].iloc[0] - SUMMARY_DF.loc[SUMMARY_DF.method == "MAMMQA_BASELINE", "EM"].iloc[0]),
    "delta_F1": float(SUMMARY_DF.loc[SUMMARY_DF.method == "CV_MAMMQA", "mean_token_F1"].iloc[0] - SUMMARY_DF.loc[SUMMARY_DF.method == "MAMMQA_BASELINE", "mean_token_F1"].iloc[0]),
    "CV_HELP_COUNT": cv_help,
    "CV_HARM_COUNT": cv_harm,
    "net_gain": cv_help - cv_harm,
    **candidate_survival,
    "CLAIM_RESCUE_COUNT": int(PER_QUESTION_DF["claim_rescue"].sum()),
    "CLAIM_DAMAGE_COUNT": int(PER_QUESTION_DF["claim_damage"].sum()),
    "repair_success_count": repair_success_count,
    "repair_candidate_rescue_count": repair_candidate_rescue,
    "repair_harm_count": repair_harm_count,
    "predictions_frozen_before_gold": True,
}
atomic_json(DIRS["metrics"] / "research_metrics.json", research_metrics)

print("\nSUMMARY")
print(SUMMARY_DF.to_string(index=False))
print("\nDISAGREEMENT ANALYSIS")
print(pd.DataFrame([disagreement_row]).to_string(index=False))
print("\nCLAIM VERIFICATION SUMMARY")
print(pd.DataFrame([claim_summary]).to_string(index=False))

# %%
# Cell 15 — required counterfactual traces for every changed final answer.
changed_rows = PER_QUESTION_DF[
    PER_QUESTION_DF["baseline_prediction"].map(normalize_answer)
    != PER_QUESTION_DF["cv_prediction"].map(normalize_answer)
]
for row in changed_rows.to_dict("records"):
    result = QUESTION_RESULTS[row["qid"]]
    print("\n" + "=" * 80)
    print("COUNTERFACTUAL:", row["qid"])
    print("Question:", row["question"])
    print("Stage-II candidates:", stage2_answers(result))
    print("Baseline final:", row["baseline_prediction"])
    print("Extracted claims and verification:")
    for candidate in result["cv_diagnostics"]["candidate_memory"]:
        print(" Candidate:", candidate["answer"], "score=", candidate["candidate_total_score"])
        for claim in candidate["claims"]:
            verification = claim.get("verification") or {}
            print("  -", claim["text"], "=>", verification.get("status"), "|", verification.get("evidence_note"))
    print("Repair:", result["cv_diagnostics"]["repair"])
    print("CV final:", row["cv_prediction"])
    print("Gold:", row["gold"])
    print("Comparison:", row["comparison_category"])

# %%
# Cell 16 — final report and cautious, automatic research interpretation.
baseline_summary = SUMMARY_DF.set_index("method").loc["MAMMQA_BASELINE"]
cv_summary = SUMMARY_DF.set_index("method").loc["CV_MAMMQA"]
unanimous_wrong = PER_QUESTION_DF[
    (PER_QUESTION_DF["stage2_agreement_type"] == "unanimous") & (~PER_QUESTION_DF["stage2_consensus_correct"])
]
verifier_failure_count = sum(
    row["failure_type"] in {"VERIFIER_FALSE_NEGATIVE", "VERIFIER_FALSE_POSITIVE"} for row in FAILURE_ROWS
)
gold_in_disagreement_pct = (
    100.0 * disagreement["gold_present_in_stage2"].mean() if len(disagreement) else math.nan
)
extra_calls = runtime_summary["average_additional_cv_calls_per_question"]

print("=" * 60)
print("CLAIM-VERIFIED MAMMQA — 16 QUESTION PILOT")
print("=" * 60)
print("Dataset: MultiModalQA dev")
print("Setting: PROVIDED CONTEXT")
print("Retrieval: NONE")
print(f"Questions: {len(PER_QUESTION_DF)}")
print("Same context for both systems: YES")
print("Same Stage-I/II outputs: YES")
print("-" * 60)
print("ORIGINAL MAMMQA")
print("-" * 60)
print(f"Exact Match: {baseline_summary.EM:.4f}")
print(f"Token F1: {baseline_summary.mean_token_F1:.4f}")
print(f"Correct: {int(baseline_summary.exact_correct)} / {len(PER_QUESTION_DF)}")
print(f"Wrong: {int(baseline_summary.incorrect_count)}")
print(f"Abstain: {int(baseline_summary.abstention_count)}")
print(f"Average logical calls/question: {baseline_summary.average_total_calls:.2f}")
print("-" * 60)
print("CV-MAMMQA")
print("-" * 60)
print(f"Exact Match: {cv_summary.EM:.4f}")
print(f"Token F1: {cv_summary.mean_token_F1:.4f}")
print(f"Correct: {int(cv_summary.exact_correct)} / {len(PER_QUESTION_DF)}")
print(f"Wrong: {int(cv_summary.incorrect_count)}")
print(f"Abstain: {int(cv_summary.abstention_count)}")
print(f"Average logical calls/question: {cv_summary.average_total_calls:.2f}")
print("-" * 60)
print("DELTA")
print("-" * 60)
print(f"ΔEM: {research_metrics['delta_EM']:+.4f}")
print(f"ΔF1: {research_metrics['delta_F1']:+.4f}")
print(f"CV helps: {cv_help}")
print(f"CV harms: {cv_harm}")
print(f"Net gain: {cv_help - cv_harm:+d}")
print("-" * 60)
print("STAGE-II DISAGREEMENT")
print("-" * 60)
print(f"Questions with disagreement: {len(disagreement)}")
print(f"Gold candidate available: {int(disagreement['gold_present_in_stage2'].sum())}")
print(f"Baseline disagreement resolution accuracy: {disagreement_row['disagreement_resolution_accuracy_baseline']:.4f}")
print(f"CV disagreement resolution accuracy: {disagreement_row['disagreement_resolution_accuracy_cv']:.4f}")
print("-" * 60)
print("CANDIDATE SURVIVAL")
print("-" * 60)
print(f"Questions with gold answer in Stage-II: {candidate_survival['questions_with_gold_in_stage2']}")
print(f"Baseline survival: {candidate_survival['baseline_candidate_survival']:.4f}")
print(f"CV survival: {candidate_survival['cv_candidate_survival']:.4f}")
print("-" * 60)
print("CLAIM VERIFICATION")
print("-" * 60)
for label in ["total_claims", "supported", "partial", "unsupported", "contradicted", "not_verifiable"]:
    print(f"{label}: {claim_summary[label]}")
print("-" * 60)
print("LOCAL REPAIR")
print("-" * 60)
print(f"Triggered: {len(repair_subset)}")
print(f"Correctly rescued: {repair_success_count}")
print(f"Candidate-level rescue: {repair_candidate_rescue}")
print(f"Harmed: {repair_harm_count}")
print(f"Repair success: {repair_success_count / len(repair_subset):.4f}" if len(repair_subset) else "Repair success: N/A")
print("-" * 60)
print("RUNTIME")
print("-" * 60)
print(f"Shared Stage-I/II: {runtime_summary['shared_stage1_stage2_total_seconds']:.1f}s")
print(f"Baseline extra: {runtime_summary['baseline_specific_total_seconds']:.1f}s")
print(f"CV extra: {runtime_summary['cv_specific_total_seconds']:.1f}s")
print(f"CV additional model calls/question: {extra_calls:.2f}")
print(f"OOM: {token_summary['oom_count']}")
print(f"Protected truncations: {token_summary['protected_truncations']}")
print(f"Deterministic evidence shortenings: {token_summary['evidence_shortenings']}")
print("-" * 60)
print("PREDICTIONS_FROZEN_BEFORE_GOLD = True")
print("PAIRED_COMPARISON_COMPLETE = True")
print("=" * 60)

print("\nRESEARCH INTERPRETATION (16-question pilot; descriptive, not a superiority claim)")
print("1. Did CV outperform baseline?", "Yes, descriptively." if research_metrics["delta_F1"] > 0 else "No descriptive F1 improvement was observed.")
print("2. Was gain mainly on disagreements?", f"Disagreement ΔEM={disagreement_row['cv_EM_disagreement'] - disagreement_row['baseline_EM_disagreement']:+.4f}; agreement ΔEM={disagreement_row['cv_EM_agreement'] - disagreement_row['baseline_EM_agreement']:+.4f}.")
print("3. Correct-candidate preservation?", f"Baseline={candidate_survival['baseline_candidate_survival']:.4f}, CV={candidate_survival['cv_candidate_survival']:.4f}.")
print("4. Local-repair rescues?", repair_success_count, "question(s); candidate-level wrong→correct:", repair_candidate_rescue)
print("5. Local-repair damage?", repair_harm_count, "baseline-correct question(s).")
print("6. Wrong unanimous Stage-II cases?", len(unanimous_wrong), list(unanimous_wrong.qid))
print("7. Verifier bottleneck?", f"Post-hoc verifier-attributed failures={verifier_failure_count}/{len(PER_QUESTION_DF)}.")
print("8. Disagreements containing a gold candidate?", f"{gold_in_disagreement_pct:.1f}%" if not math.isnan(gold_in_disagreement_pct) else "N/A")
print("9. Verification worth the cost?", f"Observed ΔF1={research_metrics['delta_F1']:+.4f} for {extra_calls:.2f} additional calls/question; interpret as a pilot trade-off.")
print("10. Next ablation:", "MAMMQA + claim verification without local repair, using these frozen qids and prompts.")
print("\nDo not modify prompts after viewing these results. N=16 supports mechanism analysis and directional evidence only.")

# %%
# Cell 17 — artifact inventory and final integrity checks.
required_artifacts = [
    DIRS["config"] / "runtime_config.json",
    DIRS["selected_questions"] / "selected_questions.json",
    PREDICTION_PATHS["shared"],
    PREDICTION_PATHS["baseline"],
    PREDICTION_PATHS["cv"],
    FREEZE_MARKER,
    DIRS["metrics"] / "per_question_comparison.csv",
    DIRS["metrics"] / "summary.csv",
    DIRS["metrics"] / "disagreement_analysis.csv",
    DIRS["metrics"] / "claim_verification_summary.csv",
    DIRS["metrics"] / "failure_analysis.csv",
    DIRS["metrics"] / "research_metrics.json",
    DIRS["metrics"] / "paired_bootstrap.json",
    DIRS["diagnostics"] / "image_manifest.json",
    DIRS["diagnostics"] / "token_cap_diagnostics.csv",
]
missing_artifacts = [str(path) for path in required_artifacts if not path.is_file()]
if missing_artifacts:
    raise RuntimeError(f"Required artifacts missing: {missing_artifacts}")
if len(PER_QUESTION_DF) != N_QUESTIONS or PER_QUESTION_DF.qid.nunique() != N_QUESTIONS:
    raise RuntimeError("Per-question metrics are not a complete paired 16-question result.")
verify_prediction_freeze()
artifact_inventory = pd.DataFrame([
    {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(OUTPUT_ROOT.rglob("*")) if path.is_file()
])
artifact_inventory.to_csv(DIRS["diagnostics"] / "artifact_inventory.csv", index=False)
print(f"Saved complete experiment to {OUTPUT_ROOT}")
print(artifact_inventory[["path", "bytes"]].to_string(index=False))
