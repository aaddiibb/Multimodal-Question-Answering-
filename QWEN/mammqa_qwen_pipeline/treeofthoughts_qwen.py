"""Tree-of-Thoughts utilities for Qwen MAMMQA experiments."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from qwen_client import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL, call_qwen, images_to_message_parts, make_content


def _clip(value: Any, max_chars: int = 10000) -> str:
    s = "" if value is None else str(value)
    return s if len(s) <= max_chars else s[:max_chars] + "\n...[truncated]"


def _parse_json_any(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def _qwen_evidence_call(
    client: Any,
    system: str,
    user_text: str,
    images: Optional[Sequence[Any]],
    text_model: str,
    vision_model: str,
    max_tokens: int = 700,
    temperature: float = 0.4,
) -> str:
    has_images = bool(images_to_message_parts(images))
    content = make_content(user_text, images) if has_images else user_text
    model = vision_model if has_images else text_model
    return call_qwen(
        client,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def generate_initial_thoughts(
    client: Any,
    question: str,
    text: str = "",
    table: str = "",
    images: Optional[Sequence[Any]] = None,
    k: int = 5,
    text_model: str = DEFAULT_TEXT_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> List[Dict[str, Any]]:
    system = """
Generate diverse initial reasoning thoughts for multimodal QA.
Each thought should be a possible evidence path, not just a final answer.
Return strict JSON: a list of objects with keys thought, evidence_needed, candidate_answer.
""".strip()
    user = f"""
Question: {question}

Text:
{_clip(text)}

Table:
{_clip(table)}

Images: attached if available.

Generate exactly {k} thoughts.
""".strip()
    out = _qwen_evidence_call(client, system, user, images, text_model, vision_model, max_tokens=900, temperature=0.7)
    parsed = _parse_json_any(out)
    if isinstance(parsed, list):
        thoughts = [x if isinstance(x, dict) else {"thought": str(x)} for x in parsed]
    else:
        lines = [ln.strip(" -0123456789.") for ln in out.splitlines() if ln.strip()]
        thoughts = [{"thought": ln} for ln in lines[:k]]
    return thoughts[:k]


def score_thought(
    client: Any,
    question: str,
    thought: Dict[str, Any] | str,
    text: str = "",
    table: str = "",
    images: Optional[Sequence[Any]] = None,
    text_model: str = DEFAULT_TEXT_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> Dict[str, Any]:
    system = """
Score the usefulness of a thought for answering the question from the provided evidence.
Return strict JSON with keys score (0 to 1) and rationale.
""".strip()
    user = f"""
Question: {question}
Thought: {json.dumps(thought, ensure_ascii=False)}

Text:
{_clip(text, 7000)}

Table:
{_clip(table, 7000)}

Images: attached if available.
""".strip()
    out = _qwen_evidence_call(client, system, user, images, text_model, vision_model, max_tokens=300, temperature=0.1)
    parsed = _parse_json_any(out)
    if isinstance(parsed, dict):
        try:
            parsed["score"] = float(parsed.get("score", 0.0))
        except Exception:
            parsed["score"] = 0.0
        parsed["raw"] = out
        return parsed
    m = re.search(r"(?:score|rating)\D+([01](?:\.\d+)?)", out, flags=re.IGNORECASE)
    return {"score": float(m.group(1)) if m else 0.0, "rationale": out, "raw": out}


def expand_thought(
    client: Any,
    question: str,
    path: List[Dict[str, Any]],
    text: str = "",
    table: str = "",
    images: Optional[Sequence[Any]] = None,
    k: int = 3,
    text_model: str = DEFAULT_TEXT_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> List[Dict[str, Any]]:
    system = """
Expand the current reasoning path with the next useful reasoning steps.
Return strict JSON: a list of objects with keys thought, operation, candidate_answer.
""".strip()
    user = f"""
Question: {question}
Current path:
{json.dumps(path, ensure_ascii=False, indent=2)}

Text:
{_clip(text, 7000)}

Table:
{_clip(table, 7000)}

Images: attached if available.

Generate exactly {k} next thoughts.
""".strip()
    out = _qwen_evidence_call(client, system, user, images, text_model, vision_model, max_tokens=800, temperature=0.6)
    parsed = _parse_json_any(out)
    if isinstance(parsed, list):
        return [x if isinstance(x, dict) else {"thought": str(x)} for x in parsed][:k]
    lines = [ln.strip(" -0123456789.") for ln in out.splitlines() if ln.strip()]
    return [{"thought": ln} for ln in lines[:k]]


def final_answer_from_path(
    client: Any,
    question: str,
    path: List[Dict[str, Any]],
    text: str = "",
    table: str = "",
    images: Optional[Sequence[Any]] = None,
    text_model: str = DEFAULT_TEXT_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> str:
    system = """
Use the selected reasoning path and evidence to produce the final answer.
Output format exactly:
<reasoning>brief reasoning</reasoning>
<answer>final answer only</answer>
""".strip()
    user = f"""
Question: {question}
Selected reasoning path:
{json.dumps(path, ensure_ascii=False, indent=2)}

Text:
{_clip(text, 8000)}

Table:
{_clip(table, 8000)}

Images: attached if available.
""".strip()
    return _qwen_evidence_call(client, system, user, images, text_model, vision_model, max_tokens=600, temperature=0.2)
