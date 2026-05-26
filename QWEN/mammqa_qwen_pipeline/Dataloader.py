"""MultimodalQA dataloader used by the Qwen MAMMQA pipeline."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


def _unwrap_kaggle_file(path: str | Path) -> Optional[str]:
    """Kaggle sometimes mounts a file as a directory containing the real file."""
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        files = [x for x in p.iterdir() if x.is_file()]
        if len(files) == 1:
            return str(files[0])
        for ext in (".jsonl", ".json"):
            hits = [x for x in files if x.name.endswith(ext)]
            if hits:
                return str(hits[0])
    return None


def find_file(root: str | Path, names: Sequence[str]) -> Optional[str]:
    root = Path(root)
    if not root.exists():
        return None
    for name in names:
        direct = _unwrap_kaggle_file(root / name)
        if direct:
            return direct
    wanted = set(names)
    for p in root.rglob("*"):
        if p.name in wanted:
            got = _unwrap_kaggle_file(p)
            if got:
                return got
    return None


def discover_mmqa_files(root: str | Path) -> Dict[str, Optional[str]]:
    return {
        "dev": find_file(root, ["MMQA_dev.jsonl", "dev.jsonl", "dev.json", "MMQA_dev.json"]),
        "texts": find_file(root, ["MMQA_texts.jsonl", "texts.jsonl", "texts.json", "MMQA_texts.json"]),
        "tables": find_file(root, ["MMQA_tables.jsonl", "tables.jsonl", "tables.json", "MMQA_tables.json"]),
        "images": find_file(root, ["MMQA_images.jsonl", "images.jsonl", "images.json", "MMQA_images.json"]),
    }


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "records", "items"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        # Fall back to dict values when it looks like an id -> record mapping.
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
    raise ValueError(f"Unsupported JSON structure in {path}")


def record_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "uid", "doc_id", "table_id", "text_doc_id", "image_doc_id", "filename", "file_name"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def make_index(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in records:
        rid = record_id(r)
        if rid:
            out[rid] = r
    return out


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    new_cols: List[str] = []
    for col in [str(c) for c in df.columns]:
        if col in counts:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


def _cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("text") or cell.get("value") or cell.get("content") or "")
    return str(cell)


def parse_table_record(record: Optional[Dict[str, Any]]) -> str:
    if not record:
        return "No table data"
    title = record.get("title") or record.get("table_title") or "Untitled table"
    table_obj = record.get("table", record)

    df: Optional[pd.DataFrame] = None
    try:
        if isinstance(table_obj, dict) and "header" in table_obj and "table_rows" in table_obj:
            headers = [_cell_text(h.get("column_name", h) if isinstance(h, dict) else h) for h in table_obj.get("header", [])]
            rows = [[_cell_text(cell) for cell in row] for row in table_obj.get("table_rows", [])]
            if rows:
                if headers and len(headers) == len(rows[0]):
                    df = pd.DataFrame(rows, columns=headers)
                else:
                    df = pd.DataFrame(rows)
        elif isinstance(table_obj, dict) and "rows" in table_obj:
            headers = table_obj.get("headers") or table_obj.get("header") or []
            rows = table_obj.get("rows") or []
            rows = [[_cell_text(cell) for cell in row] for row in rows]
            if rows:
                df = pd.DataFrame(rows, columns=headers if headers and len(headers) == len(rows[0]) else None)
        elif isinstance(table_obj, list):
            df = pd.DataFrame(table_obj)
    except Exception:
        df = None

    if df is None or df.empty:
        raw = record.get("text") or record.get("markdown") or record.get("html") or str(record)[:4000]
        return f"Table title: {title}\n{raw}"
    df = make_columns_unique(df)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_csv(index=False)
    return f"Table title: {title}\n{md}"


def format_text_record(record: Dict[str, Any]) -> str:
    title = record.get("title") or record.get("page_title") or record.get("name") or "Untitled text"
    text = record.get("text") or record.get("contents") or record.get("passage") or record.get("content") or ""
    return f"title: {title}\ntext: {text}"


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    return [str(value)]


def _answer_to_str(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, list):
        return " | ".join(str(x) for x in answer)
    if isinstance(answer, dict):
        for key in ("answer", "text", "value"):
            if key in answer:
                return _answer_to_str(answer[key])
    return str(answer)


def _file_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("utf-8")


class MultiModalQADataLoader:
    """Load MultimodalQA dev/text/table/image files and create agent inputs."""

    def __init__(
        self,
        dev_file: str,
        tables_file: str,
        texts_file: str,
        images_file: Optional[str] = None,
        images_base_url: Optional[str] = None,
        encode_images: bool = False,
    ) -> None:
        self.dev_file = _unwrap_kaggle_file(dev_file) or dev_file
        self.tables_file = _unwrap_kaggle_file(tables_file) or tables_file
        self.texts_file = _unwrap_kaggle_file(texts_file) or texts_file
        self.images_file = (_unwrap_kaggle_file(images_file) if images_file else None) or images_file
        self.images_base_url = images_base_url
        self.encode_images = encode_images

        self.dev = load_records(self.dev_file)
        self.tables = load_records(self.tables_file)
        self.texts = load_records(self.texts_file)
        self.images = load_records(self.images_file) if self.images_file else []

        self.table_lookup = make_index(self.tables)
        self.text_lookup = make_index(self.texts)
        self.image_lookup = make_index(self.images)

    @classmethod
    def from_root(cls, root: str | Path, images_base_url: Optional[str] = None, encode_images: bool = False) -> "MultiModalQADataLoader":
        root = Path(root)
        files = discover_mmqa_files(root)
        missing = [k for k in ("dev", "texts", "tables") if not files.get(k)]
        if missing:
            raise FileNotFoundError(f"Could not find required MMQA files {missing} under {root}. Found: {files}")
        if images_base_url is None:
            for candidate in (root / "images", root / "MMQA_images", root):
                if candidate.exists():
                    images_base_url = str(candidate)
                    break
        return cls(
            dev_file=files["dev"],
            tables_file=files["tables"],
            texts_file=files["texts"],
            images_file=files.get("images"),
            images_base_url=images_base_url,
            encode_images=encode_images,
        )

    def __len__(self) -> int:
        return len(self.dev)

    def _resolve_image_record(self, image_id: str) -> Optional[str]:
        rec = self.image_lookup.get(str(image_id))
        if not rec:
            return None
        for key in ("url", "image_url", "src"):
            if rec.get(key):
                return str(rec[key])
        path_value = rec.get("path") or rec.get("file_name") or rec.get("filename") or rec.get("image")
        if path_value:
            p = Path(str(path_value))
            if not p.is_absolute() and self.images_base_url:
                p = Path(self.images_base_url) / p
            if p.exists():
                return _file_to_data_url(p) if self.encode_images else str(p)
            s = str(path_value)
            if s.startswith("http://") or s.startswith("https://") or s.startswith("data:"):
                return s
        title = rec.get("title") or rec.get("page_title")
        if title:
            return f"[Image: {title}]"
        return None

    def get_agent_inputs(self, index: int) -> Dict[str, Any]:
        entry = self.dev[index]
        meta = entry.get("metadata") or entry.get("meta") or {}
        question = entry.get("question") or entry.get("query") or ""
        qid = entry.get("id") or entry.get("qid") or entry.get("question_id") or str(index)
        qtype = entry.get("type") or entry.get("question_type") or meta.get("type") or meta.get("question_type") or "unknown"

        answer = entry.get("answer")
        if answer is None:
            answer = entry.get("answers")
        if answer is None:
            answer = meta.get("answer") or meta.get("answers")

        text_ids = _as_list(meta.get("text_doc_ids") or meta.get("text_ids") or entry.get("text_doc_ids") or entry.get("text_ids"))
        table_ids = _as_list(meta.get("table_id") or meta.get("table_doc_ids") or meta.get("table_ids") or entry.get("table_id"))
        image_ids = _as_list(meta.get("image_doc_ids") or meta.get("image_ids") or entry.get("image_doc_ids") or entry.get("image_ids"))

        text_chunks = [format_text_record(self.text_lookup[tid]) for tid in text_ids if tid in self.text_lookup]
        text = "\n\n".join(text_chunks) if text_chunks else "No text available"

        table_chunks = [parse_table_record(self.table_lookup[tid]) for tid in table_ids if tid in self.table_lookup]
        table = "\n\n".join(table_chunks) if table_chunks else "No table data"

        images: List[str] = []
        for iid in image_ids:
            resolved = self._resolve_image_record(iid)
            if resolved:
                images.append(resolved)

        modalities = {
            "text": text != "No text available",
            "table": table != "No table data",
            "image": bool([x for x in images if not str(x).startswith("[Image:")]),
        }
        return {
            "id": str(qid),
            "question": question,
            "answer": _answer_to_str(answer),
            "raw_answer": answer,
            "type": qtype,
            "text": text,
            "table": table,
            "images": images,
            "metadata": meta,
            "modalities": modalities,
        }
