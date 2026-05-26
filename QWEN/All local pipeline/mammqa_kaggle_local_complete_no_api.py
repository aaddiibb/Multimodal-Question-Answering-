# Auto-extracted from mammqa_kaggle_local_complete_no_api.ipynb
# Run as a script on Kaggle only after installing dependencies.



# ================================================================================
# ============================================================
# Cell 1 — Install dependencies
# ============================================================
# Internet must be ON in Kaggle for the first run so Hugging Face models can download.
# pip install -U transformers accelerate bitsandbytes sentencepiece pillow pandas tqdm tabulate scikit-learn requests


# ================================================================================
# ============================================================
# Cell 2 — Imports and global configuration
# ============================================================
import os
import re
import gc
import json
import math
import time
import glob
import shutil
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)

print("Imports successful.")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ----------------------------
# Model configuration
# ----------------------------
# Good Kaggle GPU default. If memory fails, switch this to:
# "Qwen/Qwen2.5-1.5B-Instruct"
TEXT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
FALLBACK_TEXT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Small local VLM for image-derived evidence.
# If it fails, the pipeline falls back to image title/metadata.
VISION_MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

# ----------------------------
# Runtime configuration
# ----------------------------
USE_IMAGE_AGENT = True
FETCH_WIKI_IMAGES_IF_LOCAL_MISSING = True

# True = cheaper/faster. For TableQ only table agent, TextQ only text agent, ImageQ only image agent.
# False = use all available modalities.
REDUCE_MODALITIES_BY_QUESTION_TYPE = True

# True = closer to the paper: always run Stage 1 + Stage 2 + Stage 3 for available modalities.
# This is slower.
STRICT_PAPER_MODE = False

# Keep tiny at first. Increase after the first successful run.
RUN_N_EXAMPLES = 3
SELECT_ONE_PER_TYPE = True

# Optional baselines.
RUN_BASELINES = True

# Optional Tree-of-Thoughts. Slow on local models, so default off.
RUN_TOT = False

# Generation settings
MAX_INPUT_TOKENS = 6144
MAX_NEW_TOKENS = 220
TEMPERATURE = 0.0

# Outputs
OUTPUT_DIR = Path("/kaggle/working/mammqa_local_complete_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_CACHE_DIR = OUTPUT_DIR / "image_cache"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Output directory:", OUTPUT_DIR)


# ================================================================================
# ============================================================
# Cell 3 — Dataset path auto-discovery
# ============================================================
# This uses the path from your previous Kaggle notebook:
# /kaggle/input/datasets/adibraian/multimodalqa-dataset
#
# Kaggle sometimes stores each file as a directory containing the real file.
# The helper below unwraps that structure automatically.

KAGGLE_INPUT_CANDIDATES = [
    Path("/kaggle/input/datasets/adibraian/multimodalqa-dataset"),
    Path("/kaggle/input/multimodalqa-dataset"),
    Path("/kaggle/input"),
]

EXPECTED_FILES = {
    "dev":    ["MMQA_dev.jsonl", "dev.jsonl", "dev.json", "MMQA_dev.json"],
    "texts":  ["MMQA_texts.jsonl", "texts.jsonl", "texts.json", "MMQA_texts.json"],
    "tables": ["MMQA_tables.jsonl", "tables.jsonl", "tables.json", "MMQA_tables.json"],
    "images": ["MMQA_images.jsonl", "images.jsonl", "images.json", "MMQA_images.json"],
}

def unwrap_kaggle_file(path: Union[str, Path]) -> Optional[Path]:
    """Return an actual file path. Handles Kaggle's file-as-folder packaging."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        # 1) Same-name file inside directory: /root/MMQA_dev.jsonl/MMQA_dev.jsonl
        same_name = path / path.name
        if same_name.is_file():
            return same_name

        # 2) Single file inside directory
        children = [p for p in path.iterdir() if p.is_file()]
        if len(children) == 1:
            return children[0]

        # 3) Any json/jsonl inside directory
        for child in children:
            if child.suffix.lower() in {".jsonl", ".json"}:
                return child
    return None

def find_named_file(root: Path, candidate_names: List[str]) -> Optional[Path]:
    """Find one expected file name under root."""
    if not root.exists():
        return None

    # Direct check first
    for name in candidate_names:
        got = unwrap_kaggle_file(root / name)
        if got and got.is_file():
            return got

    # Recursive check. Keep it simple because Kaggle input dirs are usually not huge.
    for name in candidate_names:
        matches = list(root.rglob(name))
        for m in matches:
            got = unwrap_kaggle_file(m)
            if got and got.is_file():
                return got

    return None

def discover_mmqa_files() -> Tuple[Path, Dict[str, Path]]:
    """Discover the MMQA files from known Kaggle locations."""
    best_root = None
    best_files = {}

    for root in KAGGLE_INPUT_CANDIDATES:
        found = {}
        for key, names in EXPECTED_FILES.items():
            p = find_named_file(root, names)
            if p:
                found[key] = p

        if len(found) > len(best_files):
            best_root = root
            best_files = found

        if all(k in found for k in EXPECTED_FILES):
            return root, found

    missing = [k for k in EXPECTED_FILES if k not in best_files]
    raise FileNotFoundError(
        "Could not find all MMQA dataset files.\n"
        f"Best root checked: {best_root}\n"
        f"Found: {best_files}\n"
        f"Missing: {missing}\n"
        "Make sure the Kaggle dataset is attached to the notebook."
    )

DATA_ROOT, MMQA_FILES = discover_mmqa_files()

DEV_FILE = MMQA_FILES["dev"]
TEXTS_FILE = MMQA_FILES["texts"]
TABLES_FILE = MMQA_FILES["tables"]
IMAGES_FILE = MMQA_FILES["images"]

print("Dataset root:", DATA_ROOT)
print("Resolved files:")
for k, p in MMQA_FILES.items():
    size_mb = p.stat().st_size / 1024 / 1024
    print(f"  {k:7s}: {p} ({size_mb:.1f} MB)")

# Possible folders where actual image files may exist.
POSSIBLE_IMAGE_DIRS = [
    DATA_ROOT / "images",
    DATA_ROOT / "MMQA_images",
    DATA_ROOT / "image",
    DATA_ROOT,
    Path("/kaggle/input"),
]
POSSIBLE_IMAGE_DIRS = [p for p in POSSIBLE_IMAGE_DIRS if p.exists()]
print("Possible image dirs:", POSSIBLE_IMAGE_DIRS)


# ================================================================================
# ============================================================
# Cell 4 — Dataset parsing and dataloader
# ============================================================
def read_json_or_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        # Some files may be dict[id] -> object
        return list(obj.values())
    raise ValueError(f"Unsupported JSON structure in {path}")

def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    new_columns, counts = [], {}
    for col in df.columns:
        col = str(col)
        if col in counts:
            counts[col] += 1
            new_columns.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_columns.append(col)
    df.columns = new_columns
    return df

def parse_table(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Parse MMQA table object into title + markdown table."""
    title = obj.get("title", "No Title")
    table_block = obj.get("table", obj)

    # Common MMQA format
    headers = []
    for h in table_block.get("header", []) if isinstance(table_block, dict) else []:
        if isinstance(h, dict):
            headers.append(h.get("column_name", h.get("text", "")))
        else:
            headers.append(str(h))

    rows = table_block.get("table_rows", []) if isinstance(table_block, dict) else []
    formatted_rows = []
    for row in rows:
        formatted_row = []
        for cell in row:
            if isinstance(cell, dict):
                formatted_row.append(str(cell.get("text", "")))
            else:
                formatted_row.append(str(cell))
        formatted_rows.append(formatted_row)

    # Fallback generic formats
    if not formatted_rows:
        generic_rows = obj.get("rows") or obj.get("data")
        if isinstance(generic_rows, list):
            formatted_rows = generic_rows

    try:
        if formatted_rows:
            if headers and all(isinstance(r, list) for r in formatted_rows) and len(headers) == len(formatted_rows[0]):
                df = pd.DataFrame(formatted_rows, columns=headers)
            else:
                df = pd.DataFrame(formatted_rows)
            df = make_columns_unique(df)
            markdown_table = df.to_markdown(index=False)
        else:
            markdown_table = json.dumps(obj, ensure_ascii=False)[:4000]
    except Exception:
        markdown_table = json.dumps(obj, ensure_ascii=False)[:4000]

    return {
        "id": obj.get("id"),
        "title": title,
        "markdown_table": markdown_table,
        "raw": obj,
    }

def parse_text(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": obj.get("id"),
        "title": obj.get("title", "No Title"),
        "text": obj.get("text", obj.get("passage", "")),
        "raw": obj,
    }

def parse_image(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": obj.get("id"),
        "title": obj.get("title", "No Title"),
        "path": obj.get("path", obj.get("file_name", obj.get("filename", ""))),
        "url": obj.get("url", ""),
        "raw": obj,
    }

def build_lookup(rows: List[Dict[str, Any]], parser):
    lookup = {}
    for obj in rows:
        rid = obj.get("id") or obj.get("uid")
        if rid is not None:
            lookup[str(rid)] = parser(obj)
    return lookup

def get_answer_text(answer_obj: Any) -> Any:
    """Return answer in a simple comparable form."""
    if answer_obj is None:
        return ""
    if isinstance(answer_obj, str):
        return answer_obj
    if isinstance(answer_obj, (int, float)):
        return str(answer_obj)
    if isinstance(answer_obj, list):
        vals = []
        for a in answer_obj:
            if isinstance(a, dict):
                vals.append(str(a.get("answer", a.get("text", a))))
            else:
                vals.append(str(a))
        return vals
    if isinstance(answer_obj, dict):
        return str(answer_obj.get("answer", answer_obj.get("text", answer_obj)))
    return str(answer_obj)

def safe_join(items: List[str], sep="\n\n") -> str:
    items = [str(x) for x in items if x is not None and str(x).strip()]
    return sep.join(items)

print("Loading MMQA files...")
dev_data = read_json_or_jsonl(DEV_FILE)
texts_raw = read_json_or_jsonl(TEXTS_FILE)
tables_raw = read_json_or_jsonl(TABLES_FILE)
images_raw = read_json_or_jsonl(IMAGES_FILE)

text_lookup = build_lookup(texts_raw, parse_text)
table_lookup = build_lookup(tables_raw, parse_table)
image_lookup = build_lookup(images_raw, parse_image)

print(f"Loaded dev questions : {len(dev_data):,}")
print(f"Indexed texts        : {len(text_lookup):,}")
print(f"Indexed tables       : {len(table_lookup):,}")
print(f"Indexed images       : {len(image_lookup):,}")

def resolve_local_image_path(path_or_name: str) -> Optional[Path]:
    """Resolve an image path/name against likely Kaggle image folders."""
    if not path_or_name:
        return None

    s = str(path_or_name)
    if s.startswith("http://") or s.startswith("https://"):
        return None

    p = Path(s)
    if p.is_file():
        return p

    # Try relative paths under possible image dirs
    for base in POSSIBLE_IMAGE_DIRS:
        candidate = base / s
        if candidate.is_file():
            return candidate

        # If path has directories, also try basename
        candidate2 = base / Path(s).name
        if candidate2.is_file():
            return candidate2

    # Recursive basename search, only for a few common image extensions
    basename = Path(s).name
    if basename and "." in basename:
        for base in POSSIBLE_IMAGE_DIRS[:3]:
            try:
                matches = list(base.rglob(basename))
                for m in matches:
                    if m.is_file():
                        return m
            except Exception:
                pass

    return None

class MultiModalQADataLoader:
    """MMQA dataloader with the same metadata fields used by your previous notebook."""
    def __init__(self, dev, text_lookup, table_lookup, image_lookup):
        self.dev = dev
        self.text_lookup = text_lookup
        self.table_lookup = table_lookup
        self.image_lookup = image_lookup

    def __len__(self):
        return len(self.dev)

    def get_agent_inputs(self, idx: int) -> Dict[str, Any]:
        entry = self.dev[idx]
        meta = entry.get("metadata", {}) or {}

        qid = entry.get("id", entry.get("qid", str(idx)))
        question = entry.get("question", entry.get("query", ""))
        qtype = meta.get("type", entry.get("type", entry.get("question_type", "unknown")))
        modalities = meta.get("modalities", entry.get("modalities", []))

        # MMQA metadata fields
        table_id = meta.get("table_id", entry.get("table_id"))
        text_ids = meta.get("text_doc_ids", entry.get("text_doc_ids", [])) or []
        image_ids = meta.get("image_doc_ids", entry.get("image_doc_ids", [])) or []

        if isinstance(text_ids, str):
            text_ids = [text_ids]
        if isinstance(image_ids, str):
            image_ids = [image_ids]

        # Text evidence
        text_parts = []
        for tid in text_ids:
            t = self.text_lookup.get(str(tid))
            if t:
                text_parts.append(f"title: {t['title']}\ntext: {t['text']}")

        # Direct embedded fallback
        if not text_parts and entry.get("text"):
            text_parts.append(str(entry["text"]))
        combined_text = safe_join(text_parts) or ""

        # Table evidence
        table_obj = self.table_lookup.get(str(table_id)) if table_id is not None else None
        table_text = ""
        if table_obj:
            table_text = f"title: {table_obj['title']}\n{table_obj['markdown_table']}"
        elif entry.get("table"):
            table_text = str(entry["table"])

        # Image evidence metadata
        image_items = []
        for iid in image_ids:
            img = self.image_lookup.get(str(iid))
            if not img:
                continue

            local_path = resolve_local_image_path(img.get("path", ""))
            image_items.append({
                "id": img.get("id"),
                "title": img.get("title", "No Title"),
                "path": img.get("path", ""),
                "url": img.get("url", ""),
                "local_path": str(local_path) if local_path else "",
                "raw": img.get("raw", {}),
            })

        if not image_items and entry.get("images"):
            imgs = entry["images"]
            if not isinstance(imgs, list):
                imgs = [imgs]
            for im in imgs:
                if isinstance(im, dict):
                    local_path = resolve_local_image_path(im.get("path", im.get("file_name", "")))
                    image_items.append({
                        "id": im.get("id", ""),
                        "title": im.get("title", "No Title"),
                        "path": im.get("path", im.get("file_name", "")),
                        "url": im.get("url", ""),
                        "local_path": str(local_path) if local_path else "",
                        "raw": im,
                    })
                else:
                    local_path = resolve_local_image_path(str(im))
                    image_items.append({
                        "id": "",
                        "title": str(im),
                        "path": str(im),
                        "url": "",
                        "local_path": str(local_path) if local_path else "",
                        "raw": {},
                    })

        answer = get_answer_text(entry.get("answers", entry.get("answer", entry.get("gold_answer", ""))))

        return {
            "id": qid,
            "question": question,
            "type": qtype,
            "modalities": modalities,
            "text": combined_text,
            "table": table_text,
            "image_items": image_items,
            "gold_answer": answer,
            "raw": entry,
        }

dl = MultiModalQADataLoader(dev_data, text_lookup, table_lookup, image_lookup)

sample = dl.get_agent_inputs(0)
print("Sample question:", sample["question"])
print("Sample type:", sample["type"])
print("Text chars:", len(sample["text"]))
print("Table chars:", len(sample["table"]))
print("Images:", len(sample["image_items"]))
print("Gold:", sample["gold_answer"])


# ================================================================================
# ============================================================
# Cell 5 — Load local Qwen text/reasoning model
# ============================================================
def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_qwen_text_model(model_name: str, fallback_model_name: str):
    """Load Qwen locally. Uses 4-bit quantization on GPU to fit Kaggle."""
    clear_gpu_cache()
    print(f"Loading text model: {model_name}")

    tokenizer = None
    model = None

    if torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            print("Loaded text model in 4-bit.")
            return tokenizer, model
        except Exception as e:
            print("4-bit load failed for main model.")
            print("Reason:", repr(e))
            clear_gpu_cache()

    # Fallback: smaller model
    try:
        print(f"Trying fallback text model: {fallback_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, trust_remote_code=True)
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model.to("cpu")
        model.eval()
        print("Loaded fallback text model.")
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Could not load text model or fallback. Last error: {repr(e)}")

tokenizer, text_model = load_qwen_text_model(TEXT_MODEL_NAME, FALLBACK_TEXT_MODEL_NAME)

def clip_text_chars(x: Any, max_chars: int = 9000) -> str:
    s = "" if x is None else str(x)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[TRUNCATED]..."

def local_chat(messages: List[Dict[str, str]], max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE) -> str:
    """OpenAI-style chat, but generated fully locally by Qwen."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    inputs = {k: v.to(text_model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(temperature and temperature > 0),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = text_model.generate(**inputs, **gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# Quick smoke test
print(local_chat([
    {"role": "system", "content": "You answer briefly."},
    {"role": "user", "content": "Say: local model ready"}
], max_new_tokens=20))


# ================================================================================
# ============================================================
# Cell 6 — Local image evidence model and image utilities
# ============================================================
vision_processor = None
vision_model = None
IMAGE_EVIDENCE_CACHE: Dict[str, str] = {}

def load_vision_model_if_needed():
    """Lazy-load the small local VLM only when the image agent needs it."""
    global vision_processor, vision_model

    if not USE_IMAGE_AGENT:
        return None, None

    if vision_processor is not None and vision_model is not None:
        return vision_processor, vision_model

    print(f"Loading vision model: {VISION_MODEL_NAME}")
    try:
        vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME, trust_remote_code=True)
        vision_model = AutoModelForVision2Seq.from_pretrained(
            VISION_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            vision_model.to("cpu")
        vision_model.eval()
        print("Vision model loaded.")
    except Exception as e:
        print("Could not load local vision model. Image agent will use metadata only.")
        print("Reason:", repr(e))
        vision_processor = None
        vision_model = None

    return vision_processor, vision_model

def sanitize_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
    return s[:max_len] or "image"

def fetch_wikipedia_thumbnail(title: str) -> Optional[Path]:
    """Fetch a Wikipedia thumbnail for an image/title. No API key required."""
    if not FETCH_WIKI_IMAGES_IF_LOCAL_MISSING or not title or title == "No Title":
        return None

    out_path = IMAGE_CACHE_DIR / f"{sanitize_filename(title)}.jpg"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    try:
        params = {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "pithumbsize": 512,
            "format": "json",
        }
        r = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=8)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        img_url = None
        for page in pages.values():
            img_url = page.get("thumbnail", {}).get("source")
            if img_url:
                break

        if not img_url:
            return None

        img = requests.get(img_url, timeout=12)
        img.raise_for_status()
        out_path.write_bytes(img.content)
        return out_path
    except Exception:
        return None

def caption_image_with_vlm(image_path: Union[str, Path], question: str = "") -> str:
    """Use the local VLM to turn image pixels into text evidence."""
    image_path = Path(image_path)
    cache_key = f"{image_path}|||{question}"
    if cache_key in IMAGE_EVIDENCE_CACHE:
        return IMAGE_EVIDENCE_CACHE[cache_key]

    proc, model = load_vision_model_if_needed()
    if proc is None or model is None:
        return f"Image exists but local vision model is unavailable: {image_path}"

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Could not open image {image_path}: {repr(e)}"

    prompt = "Describe the image in factual terms."
    if question:
        prompt += f" Focus on details that could help answer this question: {question}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        chat_prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=chat_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        caption = proc.decode(generated, skip_special_tokens=True).strip()
    except Exception as e:
        caption = f"Vision captioning failed for {image_path}: {repr(e)}"

    IMAGE_EVIDENCE_CACHE[cache_key] = caption
    return caption

def build_image_evidence(image_items: List[Dict[str, Any]], question: str) -> str:
    """Build text evidence for images using local file, Wikipedia thumbnail, or metadata."""
    if not image_items:
        return "No image evidence available."

    pieces = []
    for idx, item in enumerate(image_items[:3], start=1):
        title = item.get("title", "No Title")
        local_path = item.get("local_path", "")
        path = item.get("path", "")
        url = item.get("url", "")

        pieces.append(f"Image {idx} metadata: title={title!r}, dataset_path={path!r}, url={url!r}")

        image_file = Path(local_path) if local_path else None
        if not image_file or not image_file.exists():
            image_file = fetch_wikipedia_thumbnail(title)

        if USE_IMAGE_AGENT and image_file and image_file.exists():
            caption = caption_image_with_vlm(image_file, question=question)
            pieces.append(f"Image {idx} visual evidence: {caption}")
        else:
            pieces.append(
                f"Image {idx} visual evidence unavailable. Use only metadata/title: {title!r}."
            )

    return "\n".join(pieces)


# ================================================================================
# ============================================================
# Cell 7 — MAMMQA agent prompts and functions
# ============================================================
BASE_SYSTEM = (
    "You are a careful question-answering agent. "
    "Use only the evidence provided in the prompt. "
    "If the evidence is insufficient, say that it is insufficient. "
    "Do not use outside knowledge."
)

def text_agent(question: str, text: str) -> str:
    messages = [
        {"role": "system", "content": BASE_SYSTEM + " You are the Stage 1 TEXT specialist."},
        {"role": "user", "content": f"""
Question:
{question}

Text evidence:
{clip_text_chars(text)}

Task:
Extract only the text evidence relevant to the question.
Then give a tentative answer if the text supports one.

Return:
Modality: Text
Evidence: <short evidence>
Tentative Answer: <answer or insufficient evidence>
""".strip()},
    ]
    return local_chat(messages)

def table_agent(question: str, table: str) -> str:
    messages = [
        {"role": "system", "content": BASE_SYSTEM + " You are the Stage 1 TABLE specialist."},
        {"role": "user", "content": f"""
Question:
{question}

Table evidence:
{clip_text_chars(table)}

Task:
Find the relevant row(s), column(s), and cell(s).
Then give a tentative answer if the table supports one.

Return:
Modality: Table
Evidence: <relevant row/column/cell information>
Tentative Answer: <answer or insufficient evidence>
""".strip()},
    ]
    return local_chat(messages)

def image_agent(question: str, image_items: List[Dict[str, Any]]) -> str:
    image_evidence = build_image_evidence(image_items, question)
    messages = [
        {"role": "system", "content": BASE_SYSTEM + " You are the Stage 1 IMAGE specialist."},
        {"role": "user", "content": f"""
Question:
{question}

Image-derived evidence:
{clip_text_chars(image_evidence)}

Task:
Use the image-derived evidence and metadata.
Then give a tentative answer if the image evidence supports one.

Return:
Modality: Image
Evidence: <visual/metadata evidence>
Tentative Answer: <answer or insufficient evidence>
""".strip()},
    ]
    return local_chat(messages)

def cross_modal_refinement_agent(
    question: str,
    anchor_modality: str,
    anchor_insight: str,
    other_insights: Dict[str, str],
) -> str:
    messages = [
        {"role": "system", "content": BASE_SYSTEM + f" You are the Stage 2 cross-modal refinement agent anchored on {anchor_modality}."},
        {"role": "user", "content": f"""
Question:
{question}

Anchor modality:
{anchor_modality}

Anchor insight:
{clip_text_chars(anchor_insight, 3500)}

Other modality insights:
{clip_text_chars(json.dumps(other_insights, ensure_ascii=False, indent=2), 5000)}

Task:
Check whether the other modalities support, contradict, or refine the anchor insight.
Return a corrected candidate answer if possible.

Return:
Anchor: {anchor_modality}
Support/Contradiction: <supported / contradicted / uncertain>
Refined Evidence: <short evidence>
Candidate Answer: <answer or insufficient evidence>
""".strip()},
    ]
    return local_chat(messages)

def aggregator_agent(question: str, stage1: Dict[str, str], stage2: Dict[str, str]) -> str:
    messages = [
        {"role": "system", "content": BASE_SYSTEM + " You are the Stage 3 final aggregator."},
        {"role": "user", "content": f"""
Question:
{question}

Stage 1 specialist outputs:
{clip_text_chars(json.dumps(stage1, ensure_ascii=False, indent=2), 6000)}

Stage 2 cross-modal refined outputs:
{clip_text_chars(json.dumps(stage2, ensure_ascii=False, indent=2), 6000)}

Task:
Resolve conflicts and give the final answer.
Return exactly this format:

Final Answer: <short answer>
Evidence: <one short sentence explaining the supporting evidence>
""".strip()},
    ]
    return local_chat(messages, max_new_tokens=160)

def extract_final_answer(raw: str) -> str:
    if raw is None:
        return ""
    raw = str(raw).strip()
    m = re.search(r"Final Answer\s*:\s*(.+)", raw, flags=re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        ans = ans.split("\n")[0].strip()
        return ans
    # Fallback: first nonempty line
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


# ================================================================================
# ============================================================
# Cell 8 — Full local MAMMQA pipeline
# ============================================================
def select_modalities_for_question(qtype: str, text: str, table: str, image_items: List[Dict[str, Any]]) -> Dict[str, bool]:
    available = {
        "text": bool(str(text).strip()),
        "table": bool(str(table).strip()),
        "image": bool(image_items),
    }

    if STRICT_PAPER_MODE or not REDUCE_MODALITIES_BY_QUESTION_TYPE:
        return available

    qt = str(qtype).lower()
    if qt == "textq":
        return {"text": available["text"], "table": False, "image": False}
    if qt == "tableq":
        return {"text": False, "table": available["table"], "image": False}
    if qt == "imageq":
        return {"text": False, "table": False, "image": available["image"]}

    # Compose/multi-hop/unknown: use all available modalities.
    return available

def get_answer_MM_local(
    question: str,
    text: str = "",
    table: str = "",
    image_items: Optional[List[Dict[str, Any]]] = None,
    qtype: str = "",
    verbose: bool = True,
) -> Dict[str, Any]:
    image_items = image_items or []
    use = select_modalities_for_question(qtype, text, table, image_items)

    if verbose:
        print("Available modalities:", {
            "text": bool(text.strip()),
            "table": bool(table.strip()),
            "image": bool(image_items),
        })
        print("Using modalities:", use)
        print("\n▶ STAGE 1 — Modality specialist agents")

    stage1 = {}

    if use["text"] or STRICT_PAPER_MODE:
        stage1["text"] = text_agent(question, text) if text.strip() else "No text evidence."
        if verbose:
            print("  Done: text specialist")

    if use["table"] or STRICT_PAPER_MODE:
        stage1["table"] = table_agent(question, table) if table.strip() else "No table evidence."
        if verbose:
            print("  Done: table specialist")

    if use["image"] or STRICT_PAPER_MODE:
        stage1["image"] = image_agent(question, image_items) if image_items else "No image evidence."
        if verbose:
            print("  Done: image specialist")

    if not stage1:
        stage1["none"] = "No usable evidence was available."

    if verbose:
        print("\n▶ STAGE 2 — Cross-modal refinement agents")

    stage2 = {}
    if STRICT_PAPER_MODE or len(stage1) > 1:
        for modality, insight in stage1.items():
            others = {k: v for k, v in stage1.items() if k != modality}
            stage2[modality] = cross_modal_refinement_agent(question, modality, insight, others)
            if verbose:
                print(f"  Done: {modality} cross-modal refinement")
    else:
        # Single-modality reduced mode: keep Stage 2 object, but avoid another expensive model call.
        only_modality = next(iter(stage1))
        stage2[only_modality] = (
            "Single-modality reduced run. No cross-modal contradiction checked. "
            f"Carry forward Stage 1 insight: {stage1[only_modality]}"
        )
        if verbose:
            print("  Reduced mode: carried single-modality evidence forward")

    if verbose:
        print("\n▶ STAGE 3 — Aggregator agent")

    final_raw = aggregator_agent(question, stage1, stage2)
    final_answer = extract_final_answer(final_raw)

    if verbose:
        print("  Done: aggregator")
        print("  Final Answer:", final_answer)

    return {
        "question": question,
        "type": qtype,
        "used_modalities": use,
        "stage1": stage1,
        "stage2": stage2,
        "final_raw": final_raw,
        "final_answer": final_answer,
    }


# ================================================================================
# ============================================================
# Cell 9 — Baselines and optional lightweight Tree-of-Thoughts
# ============================================================
def get_answer_zs_no_data_local(question: str) -> Dict[str, str]:
    messages = [
        {"role": "system", "content": "Answer the question directly and concisely. If unknown, say insufficient evidence."},
        {"role": "user", "content": f"Question: {question}\n\nReturn exactly:\nFinal Answer: <answer>"},
    ]
    raw = local_chat(messages, max_new_tokens=90)
    return {"final_raw": raw, "final_answer": extract_final_answer(raw)}

def get_answer_cot_local(question: str, text: str, table: str, image_items: List[Dict[str, Any]]) -> Dict[str, str]:
    image_evidence = build_image_evidence(image_items, question) if image_items else "No image evidence."
    messages = [
        {"role": "system", "content": BASE_SYSTEM + " You are a single-agent multimodal baseline."},
        {"role": "user", "content": f"""
Question:
{question}

Text evidence:
{clip_text_chars(text, 3500)}

Table evidence:
{clip_text_chars(table, 3500)}

Image-derived evidence:
{clip_text_chars(image_evidence, 2500)}

Task:
Use the evidence and give a concise answer.
Return exactly:

Final Answer: <short answer>
Evidence: <one short sentence>
""".strip()},
    ]
    raw = local_chat(messages, max_new_tokens=160)
    return {"final_raw": raw, "final_answer": extract_final_answer(raw)}

def generate_initial_thoughts(question: str, evidence: str, k: int = 3) -> List[str]:
    messages = [
        {"role": "system", "content": BASE_SYSTEM + " Generate diverse candidate solution paths, not final long reasoning."},
        {"role": "user", "content": f"""
Question:
{question}

Evidence:
{clip_text_chars(evidence, 5000)}

Generate {k} short candidate thoughts. Number them 1 to {k}.
""".strip()},
    ]
    raw = local_chat(messages, max_new_tokens=220)
    thoughts = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*\d+[\).\-\:]\s*", "", line).strip()
        if line:
            thoughts.append(line)
    return thoughts[:k] if thoughts else [raw.strip()]

def score_thought(question: str, evidence: str, thought: str) -> float:
    messages = [
        {"role": "system", "content": BASE_SYSTEM + " Score the candidate from 0 to 1. Return only a number."},
        {"role": "user", "content": f"""
Question:
{question}

Evidence:
{clip_text_chars(evidence, 4000)}

Candidate thought:
{thought}

Score how likely this thought leads to the correct answer, from 0 to 1.
Return only the number.
""".strip()},
    ]
    raw = local_chat(messages, max_new_tokens=20)
    m = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", raw)
    return float(m.group(0)) if m else 0.0

def run_lightweight_tot(question: str, text: str, table: str, image_items: List[Dict[str, Any]], k: int = 3) -> Dict[str, Any]:
    image_evidence = build_image_evidence(image_items, question) if image_items else ""
    evidence = f"TEXT:\n{text}\n\nTABLE:\n{table}\n\nIMAGE:\n{image_evidence}"
    thoughts = generate_initial_thoughts(question, evidence, k=k)
    scored = [(t, score_thought(question, evidence, t)) for t in thoughts]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_thought = scored[0][0] if scored else ""

    messages = [
        {"role": "system", "content": BASE_SYSTEM + " Use the best candidate thought to answer concisely."},
        {"role": "user", "content": f"""
Question:
{question}

Evidence:
{clip_text_chars(evidence, 6000)}

Best candidate thought:
{best_thought}

Return exactly:
Final Answer: <short answer>
Evidence: <one short sentence>
""".strip()},
    ]
    raw = local_chat(messages, max_new_tokens=160)
    return {
        "thoughts": scored,
        "final_raw": raw,
        "final_answer": extract_final_answer(raw),
    }


# ================================================================================
# ============================================================
# Cell 10 — Evaluation utilities
# ============================================================
def normalize_answer(s: Any) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: Any, gold: Any) -> int:
    if isinstance(gold, list):
        return int(any(exact_match(pred, g) for g in gold))
    return int(normalize_answer(pred) == normalize_answer(gold))

def token_f1_single(pred: Any, gold: Any) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    counts = {}
    for tok in p:
        counts[tok] = counts.get(tok, 0) + 1

    overlap = 0
    for tok in g:
        if counts.get(tok, 0) > 0:
            overlap += 1
            counts[tok] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)

def token_f1(pred: Any, gold: Any) -> float:
    if isinstance(gold, list):
        return max(token_f1_single(pred, g) for g in gold) if gold else 0.0
    return token_f1_single(pred, gold)

def evaluate_prediction_rows(rows: List[Dict[str, Any]], prediction_key: str = "prediction") -> Dict[str, float]:
    if not rows:
        return {"n": 0, "exact_match": 0.0, "token_f1": 0.0}

    ems = [exact_match(r.get(prediction_key, ""), r.get("gold_answer", "")) for r in rows]
    f1s = [token_f1(r.get(prediction_key, ""), r.get("gold_answer", "")) for r in rows]
    return {
        "n": len(rows),
        "exact_match": sum(ems) / len(ems),
        "token_f1": sum(f1s) / len(f1s),
    }

def save_jsonl(rows: List[Dict[str, Any]], path: Union[str, Path]):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ================================================================================
# ============================================================
# Cell 11 — Select examples
# ============================================================
def select_example_indices(dl: MultiModalQADataLoader, n: int, one_per_type: bool = True) -> List[int]:
    if not one_per_type:
        return list(range(min(n, len(dl))))

    target_types = ["TextQ", "TableQ", "ImageQ", "Compose"]
    selected = []
    seen = set()

    for i, entry in enumerate(dl.dev):
        qtype = (entry.get("metadata", {}) or {}).get("type", entry.get("type", "unknown"))
        if qtype in target_types and qtype not in seen:
            selected.append(i)
            seen.add(qtype)
        if len(selected) >= n:
            break

    # Fill remaining with first examples not already selected
    if len(selected) < n:
        for i in range(len(dl)):
            if i not in selected:
                selected.append(i)
            if len(selected) >= n:
                break

    return selected

selected_indices = select_example_indices(dl, RUN_N_EXAMPLES, SELECT_ONE_PER_TYPE)
print("Selected indices:", selected_indices)
for j, idx in enumerate(selected_indices, 1):
    ex = dl.get_agent_inputs(idx)
    print(f"{j}. idx={idx}, type={ex['type']}, question={ex['question'][:100]}")


# ================================================================================
# ============================================================
# Cell 12 — Run local MAMMQA pipeline
# ============================================================
mammqa_rows = []

for run_i, idx in enumerate(tqdm(selected_indices), start=1):
    ex = dl.get_agent_inputs(idx)

    print("\n" + "#" * 70)
    print(f"EXAMPLE {run_i}/{len(selected_indices)} | idx={idx} | Type: {ex['type']}")
    print("QUESTION:", ex["question"])
    print("GOLD:", ex["gold_answer"])

    result = get_answer_MM_local(
        question=ex["question"],
        text=ex["text"],
        table=ex["table"],
        image_items=ex["image_items"],
        qtype=ex["type"],
        verbose=True,
    )

    row = {
        "id": ex["id"],
        "dataset_index": idx,
        "type": ex["type"],
        "modalities": ex["modalities"],
        "question": ex["question"],
        "gold_answer": ex["gold_answer"],
        "prediction": result["final_answer"],
        "used_modalities": result["used_modalities"],
        "final_raw": result["final_raw"],
        "stage1": result["stage1"],
        "stage2": result["stage2"],
    }
    row["exact_match"] = exact_match(row["prediction"], row["gold_answer"])
    row["token_f1"] = token_f1(row["prediction"], row["gold_answer"])
    mammqa_rows.append(row)

    print("PRED:", row["prediction"])
    print("EM:", row["exact_match"], "F1:", round(row["token_f1"], 3))

# Save outputs
mammqa_jsonl = OUTPUT_DIR / "mammqa_local_predictions.jsonl"
mammqa_csv = OUTPUT_DIR / "mammqa_local_predictions.csv"

save_jsonl(mammqa_rows, mammqa_jsonl)

# Flatten nested fields for CSV
csv_rows = []
for r in mammqa_rows:
    flat = dict(r)
    flat["gold_answer"] = json.dumps(flat["gold_answer"], ensure_ascii=False) if isinstance(flat["gold_answer"], list) else flat["gold_answer"]
    flat["modalities"] = json.dumps(flat["modalities"], ensure_ascii=False)
    flat["used_modalities"] = json.dumps(flat["used_modalities"], ensure_ascii=False)
    flat["stage1"] = json.dumps(flat["stage1"], ensure_ascii=False)
    flat["stage2"] = json.dumps(flat["stage2"], ensure_ascii=False)
    csv_rows.append(flat)

pd.DataFrame(csv_rows).to_csv(mammqa_csv, index=False)

print("\nSaved:", mammqa_jsonl)
print("Saved:", mammqa_csv)
print("MAMMQA eval:", evaluate_prediction_rows(mammqa_rows))


# ================================================================================
# ============================================================
# Cell 13 — Run baselines
# ============================================================
baseline_rows = []

if RUN_BASELINES:
    for run_i, idx in enumerate(tqdm(selected_indices), start=1):
        ex = dl.get_agent_inputs(idx)
        print("\n" + "-" * 70)
        print(f"BASELINES {run_i}/{len(selected_indices)} | idx={idx} | Type: {ex['type']}")
        print("QUESTION:", ex["question"])

        zs = get_answer_zs_no_data_local(ex["question"])
        cot = get_answer_cot_local(ex["question"], ex["text"], ex["table"], ex["image_items"])

        row = {
            "id": ex["id"],
            "dataset_index": idx,
            "type": ex["type"],
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "zero_shot_prediction": zs["final_answer"],
            "zero_shot_raw": zs["final_raw"],
            "cot_prediction": cot["final_answer"],
            "cot_raw": cot["final_raw"],
        }
        baseline_rows.append(row)

        print("ZS :", row["zero_shot_prediction"])
        print("CoT:", row["cot_prediction"])

    baseline_jsonl = OUTPUT_DIR / "local_baselines.jsonl"
    baseline_csv = OUTPUT_DIR / "local_baselines.csv"
    save_jsonl(baseline_rows, baseline_jsonl)

    baseline_csv_rows = []
    for r in baseline_rows:
        flat = dict(r)
        flat["gold_answer"] = json.dumps(flat["gold_answer"], ensure_ascii=False) if isinstance(flat["gold_answer"], list) else flat["gold_answer"]
        baseline_csv_rows.append(flat)
    pd.DataFrame(baseline_csv_rows).to_csv(baseline_csv, index=False)

    zs_eval_rows = [{"prediction": r["zero_shot_prediction"], "gold_answer": r["gold_answer"]} for r in baseline_rows]
    cot_eval_rows = [{"prediction": r["cot_prediction"], "gold_answer": r["gold_answer"]} for r in baseline_rows]

    print("\nSaved:", baseline_jsonl)
    print("Saved:", baseline_csv)
    print("Zero-shot eval:", evaluate_prediction_rows(zs_eval_rows))
    print("CoT eval:", evaluate_prediction_rows(cot_eval_rows))
else:
    print("RUN_BASELINES=False, skipped.")


# ================================================================================
# ============================================================
# Cell 14 — Optional lightweight Tree-of-Thoughts
# ============================================================
tot_rows = []

if RUN_TOT:
    for run_i, idx in enumerate(tqdm(selected_indices), start=1):
        ex = dl.get_agent_inputs(idx)
        print("\n" + "=" * 70)
        print(f"TOT {run_i}/{len(selected_indices)} | idx={idx} | Type: {ex['type']}")
        print("QUESTION:", ex["question"])

        tot = run_lightweight_tot(
            question=ex["question"],
            text=ex["text"],
            table=ex["table"],
            image_items=ex["image_items"],
            k=3,
        )

        row = {
            "id": ex["id"],
            "dataset_index": idx,
            "type": ex["type"],
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "tot_prediction": tot["final_answer"],
            "tot_raw": tot["final_raw"],
            "thoughts": tot["thoughts"],
        }
        row["exact_match"] = exact_match(row["tot_prediction"], row["gold_answer"])
        row["token_f1"] = token_f1(row["tot_prediction"], row["gold_answer"])
        tot_rows.append(row)

        print("ToT:", row["tot_prediction"])

    tot_jsonl = OUTPUT_DIR / "local_tot.jsonl"
    tot_csv = OUTPUT_DIR / "local_tot.csv"
    save_jsonl(tot_rows, tot_jsonl)

    tot_csv_rows = []
    for r in tot_rows:
        flat = dict(r)
        flat["gold_answer"] = json.dumps(flat["gold_answer"], ensure_ascii=False) if isinstance(flat["gold_answer"], list) else flat["gold_answer"]
        flat["thoughts"] = json.dumps(flat["thoughts"], ensure_ascii=False)
        tot_csv_rows.append(flat)
    pd.DataFrame(tot_csv_rows).to_csv(tot_csv, index=False)

    print("\nSaved:", tot_jsonl)
    print("Saved:", tot_csv)
    print("ToT eval:", evaluate_prediction_rows([{"prediction": r["tot_prediction"], "gold_answer": r["gold_answer"]} for r in tot_rows]))
else:
    print("RUN_TOT=False, skipped.")
