"""Qwen implementation of MAMMQA agents, baselines, and experiment runner."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from qwen_client import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL, call_qwen, images_to_message_parts, make_content


TEMPERATURE = 0.2
TOP_P = 0.8
MAX_TOKENS = 700


def _clip(value: Any, max_chars: int = 12000) -> str:
    s = "" if value is None else str(value)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n...[truncated {len(s) - max_chars} chars]"


def extract_answer_tag(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", str(text), flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Also support JSON-like outputs.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip()
    except Exception:
        pass
    return str(text).strip()


AGENT_STAGE1_SYSTEM_PROMPT = """
You are a modality-specialist agent for multimodal question answering.
You receive ONE modality plus a question. Do not guess beyond the provided input.

Tasks:
1. Identify the modality: text, table, or image.
2. Extract only evidence relevant to the question.
3. Break the question into smaller sub-questions if useful.
4. State whether this modality is sufficient, insufficient, or contradictory.
5. Give a short candidate answer only if directly supported.

Output format:
<modality>...</modality>
<evidence>bullet list of grounded evidence</evidence>
<subquestions>bullet list if needed</subquestions>
<candidate_answer>answer or NOT_ENOUGH_INFORMATION</candidate_answer>
<confidence>low/medium/high</confidence>
""".strip()


AGENT_STAGE2_SYSTEM_PROMPT = """
You are a cross-modal refinement agent.
You receive one modality-specialist insight as the anchor, plus other available raw modalities.

Tasks:
1. Verify the anchor insight against the other modalities.
2. Add missing evidence from the other modalities.
3. Resolve conflicts and say which evidence is strongest.
4. Produce a refined candidate answer.

Use only the supplied evidence. If evidence is missing, say so.

Output format:
<verified_evidence>...</verified_evidence>
<conflicts>...</conflicts>
<refined_reasoning>...</refined_reasoning>
<candidate_answer>answer or NOT_ENOUGH_INFORMATION</candidate_answer>
<confidence>low/medium/high</confidence>
""".strip()


AGGREGATOR_SYSTEM_PROMPT = """
You are the final MAMMQA aggregator agent.
You receive outputs from modality specialists and cross-modal refinement agents.

Tasks:
1. Compare all candidate answers and confidence levels.
2. Prefer answers grounded in explicit evidence.
3. Resolve contradictions conservatively.
4. Return a concise final answer.

Output format exactly:
<reasoning>brief evidence-based reasoning</reasoning>
<answer>final answer only</answer>
""".strip()


ZS_SYSTEM_PROMPT = """
Answer the question directly. If you are unsure, give the best concise answer.
Output format exactly:
<answer>final answer only</answer>
""".strip()


COT_SYSTEM_PROMPT = """
You are given a question and specific multimodal evidence. Answer using ONLY the provided evidence.
Think step by step, then give the answer.
Output format exactly:
<reasoning>brief reasoning grounded in the evidence</reasoning>
<answer>final answer only</answer>
""".strip()


def _call_text(client: Any, system: str, user: str, model: Optional[str] = None, **kwargs: Any) -> str:
    return call_qwen(
        client,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model or DEFAULT_TEXT_MODEL,
        temperature=kwargs.pop("temperature", TEMPERATURE),
        top_p=kwargs.pop("top_p", TOP_P),
        max_tokens=kwargs.pop("max_tokens", MAX_TOKENS),
        **kwargs,
    )


def _call_maybe_vision(
    client: Any,
    system: str,
    user_text: str,
    images: Optional[Sequence[Any]] = None,
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    **kwargs: Any,
) -> str:
    image_parts = images_to_message_parts(images)
    content = make_content(user_text, images) if image_parts else user_text
    model = (vision_model or DEFAULT_VISION_MODEL) if image_parts else (text_model or DEFAULT_TEXT_MODEL)
    return call_qwen(
        client,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        model=model,
        temperature=kwargs.pop("temperature", TEMPERATURE),
        top_p=kwargs.pop("top_p", TOP_P),
        max_tokens=kwargs.pop("max_tokens", MAX_TOKENS),
        **kwargs,
    )


# ----------------------------- Stage 1 agents -----------------------------

def text_agent(client: Any, question: str, texts: str, model: Optional[str] = None, **kwargs: Any) -> str:
    if not texts or str(texts).strip().lower() == "no text available":
        return "NO_TEXT_AVAILABLE: No text modality was provided."
    user = f"Question:\n{question}\n\nText modality:\n{_clip(texts)}"
    return _call_text(client, AGENT_STAGE1_SYSTEM_PROMPT, user, model=model, **kwargs)


def table_agent(client: Any, question: str, tables: str, model: Optional[str] = None, **kwargs: Any) -> str:
    if not tables or str(tables).strip().lower() == "no table data":
        return "NO_TABLE_AVAILABLE: No table modality was provided."
    user = f"Question:\n{question}\n\nTable modality as markdown:\n{_clip(tables)}"
    return _call_text(client, AGENT_STAGE1_SYSTEM_PROMPT, user, model=model, **kwargs)


def image_agent(client: Any, question: str, images: Optional[Sequence[Any]], model: Optional[str] = None, **kwargs: Any) -> str:
    if not images_to_message_parts(images):
        return "NO_IMAGE_AVAILABLE: No usable image modality was provided."
    user = f"Question:\n{question}\n\nImage modality: the attached image(s). Extract visual evidence relevant to the question."
    return _call_maybe_vision(client, AGENT_STAGE1_SYSTEM_PROMPT, user, images=images, vision_model=model, **kwargs)


# ----------------------------- Stage 2 agents -----------------------------

def text_cross_agent(
    client: Any,
    question: str,
    text_insight: str,
    raw_tables: str,
    raw_images: Optional[Sequence[Any]],
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    **kwargs: Any,
) -> str:
    if text_insight.startswith("NO_TEXT_AVAILABLE"):
        return "TEXT_CROSS_SKIPPED: No text insight exists to refine."
    user = f"""
Question:
{question}

Anchor insight from text specialist:
{_clip(text_insight, 8000)}

Other modality - table:
{_clip(raw_tables)}

Other modality - image(s): attached if available.
""".strip()
    return _call_maybe_vision(client, AGENT_STAGE2_SYSTEM_PROMPT, user, images=raw_images, text_model=text_model, vision_model=vision_model, **kwargs)


def table_cross_agent(
    client: Any,
    question: str,
    table_insight: str,
    raw_text: str,
    raw_images: Optional[Sequence[Any]],
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    **kwargs: Any,
) -> str:
    if table_insight.startswith("NO_TABLE_AVAILABLE"):
        return "TABLE_CROSS_SKIPPED: No table insight exists to refine."
    user = f"""
Question:
{question}

Anchor insight from table specialist:
{_clip(table_insight, 8000)}

Other modality - text:
{_clip(raw_text)}

Other modality - image(s): attached if available.
""".strip()
    return _call_maybe_vision(client, AGENT_STAGE2_SYSTEM_PROMPT, user, images=raw_images, text_model=text_model, vision_model=vision_model, **kwargs)


def image_cross_agent(
    client: Any,
    question: str,
    image_insight: str,
    raw_text: str,
    raw_tables: str,
    raw_images: Optional[Sequence[Any]],
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    **kwargs: Any,
) -> str:
    if image_insight.startswith("NO_IMAGE_AVAILABLE"):
        return "IMAGE_CROSS_SKIPPED: No image insight exists to refine."
    user = f"""
Question:
{question}

Anchor insight from image specialist:
{_clip(image_insight, 8000)}

Other modality - text:
{_clip(raw_text)}

Other modality - table:
{_clip(raw_tables)}

Original image(s): attached again if available.
""".strip()
    return _call_maybe_vision(client, AGENT_STAGE2_SYSTEM_PROMPT, user, images=raw_images, text_model=text_model, vision_model=vision_model, **kwargs)


# ----------------------------- Stage 3 aggregator -----------------------------

def reasoning_agent(
    client: Any,
    question: str,
    text_insight: str,
    table_insight: str,
    image_insight: str,
    text_cross: str,
    table_cross: str,
    image_cross: str,
    model: Optional[str] = None,
    **kwargs: Any,
) -> str:
    user = f"""
Question:
{question}

STAGE 1 - Text specialist:
{_clip(text_insight, 6000)}

STAGE 1 - Table specialist:
{_clip(table_insight, 6000)}

STAGE 1 - Image specialist:
{_clip(image_insight, 6000)}

STAGE 2 - Text-anchored refinement:
{_clip(text_cross, 6000)}

STAGE 2 - Table-anchored refinement:
{_clip(table_cross, 6000)}

STAGE 2 - Image-anchored refinement:
{_clip(image_cross, 6000)}
""".strip()
    return _call_text(client, AGGREGATOR_SYSTEM_PROMPT, user, model=model, max_tokens=kwargs.pop("max_tokens", 500), **kwargs)


def _available_modalities(text: str, tables: str, images: Optional[Sequence[Any]]) -> Dict[str, bool]:
    return {
        "text": bool(text and str(text).strip().lower() != "no text available"),
        "table": bool(tables and str(tables).strip().lower() != "no table data"),
        "image": bool(images_to_message_parts(images)),
    }


def _needed_by_qtype(qtype: Optional[str]) -> Optional[set[str]]:
    if not qtype:
        return None
    q = str(qtype).lower()
    if "compose" in q or "multi" in q or "+" in q:
        return {"text", "table", "image"}
    if "table" in q:
        return {"table"}
    if "image" in q:
        return {"image"}
    if "text" in q:
        return {"text"}
    return None


def get_answer_MM(
    client: Any,
    question: str,
    text: str,
    tables: str,
    images: Optional[Sequence[Any]],
    model: Optional[str] = None,
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    qtype: Optional[str] = None,
    strict_paper_mode: bool = True,
    reduce_modalities_by_question_type: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, str]:
    """Run the complete three-stage MAMMQA pipeline.

    strict_paper_mode=True keeps Stage 2 and Stage 3 even for single-modality
    questions. reduce_modalities_by_question_type=True is a cost-saving mode.
    """
    text_model = text_model or model or DEFAULT_TEXT_MODEL
    vision_model = vision_model or DEFAULT_VISION_MODEL
    avail = _available_modalities(text, tables, images)
    needed = set(avail.keys())
    if reduce_modalities_by_question_type:
        by_type = _needed_by_qtype(qtype)
        if by_type:
            needed = by_type
    # Never attempt unavailable modalities.
    needed = {m for m in needed if avail.get(m)}
    if strict_paper_mode and not needed:
        needed = {m for m, ok in avail.items() if ok}

    if verbose:
        print(f"Available modalities: {avail}; running specialists for: {sorted(needed)}")
        print("Stage 1: modality specialists")

    text_insight = text_agent(client, question, text, model=text_model, **kwargs) if "text" in needed else "NO_TEXT_AVAILABLE: Text specialist not run."
    if verbose and "text" in needed: print("  done text specialist")
    table_insight = table_agent(client, question, tables, model=text_model, **kwargs) if "table" in needed else "NO_TABLE_AVAILABLE: Table specialist not run."
    if verbose and "table" in needed: print("  done table specialist")
    image_insight = image_agent(client, question, images, model=vision_model, **kwargs) if "image" in needed else "NO_IMAGE_AVAILABLE: Image specialist not run."
    if verbose and "image" in needed: print("  done image specialist")

    if verbose:
        print("Stage 2: cross-modal refinement")
    # Full paper mode still performs refinement for every modality that has a Stage 1 insight.
    text_cross = text_cross_agent(client, question, text_insight, tables, images, text_model=text_model, vision_model=vision_model, **kwargs) if "text" in needed else "TEXT_CROSS_SKIPPED: Text specialist not run."
    if verbose and "text" in needed: print("  done text-anchored refinement")
    table_cross = table_cross_agent(client, question, table_insight, text, images, text_model=text_model, vision_model=vision_model, **kwargs) if "table" in needed else "TABLE_CROSS_SKIPPED: Table specialist not run."
    if verbose and "table" in needed: print("  done table-anchored refinement")
    image_cross = image_cross_agent(client, question, image_insight, text, tables, images, text_model=text_model, vision_model=vision_model, **kwargs) if "image" in needed else "IMAGE_CROSS_SKIPPED: Image specialist not run."
    if verbose and "image" in needed: print("  done image-anchored refinement")

    if verbose:
        print("Stage 3: aggregator")
    final = reasoning_agent(
        client,
        question,
        text_insight,
        table_insight,
        image_insight,
        text_cross,
        table_cross,
        image_cross,
        model=text_model,
        **kwargs,
    )

    return {
        "Text Agent Output": text_insight,
        "Table Agent Output": table_insight,
        "Image Agent Output": image_insight,
        "Text Cross Agent Output": text_cross,
        "Table Cross Agent Output": table_cross,
        "Image Cross Agent Output": image_cross,
        "Final Answer": final,
        "predicted_answer": extract_answer_tag(final),
        "available_modalities": json.dumps(avail),
        "ran_modalities": json.dumps(sorted(needed)),
    }


# ----------------------------- Baselines -----------------------------

def get_answer_zs_no_data(client: Any, question: str, model: Optional[str] = None, **kwargs: Any) -> str:
    return _call_text(client, ZS_SYSTEM_PROMPT, f"Question:\n{question}", model=model or DEFAULT_TEXT_MODEL, max_tokens=kwargs.pop("max_tokens", 250), **kwargs)


def get_answer_cot(
    client: Any,
    question: str,
    text: str,
    table: str,
    images: Optional[Sequence[Any]],
    model: Optional[str] = None,
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    **kwargs: Any,
) -> str:
    user = f"""
Question:
{question}

Text evidence:
{_clip(text)}

Table evidence:
{_clip(table)}

Image evidence: attached if available.
""".strip()
    return _call_maybe_vision(
        client,
        COT_SYSTEM_PROMPT,
        user,
        images=images,
        text_model=text_model or model or DEFAULT_TEXT_MODEL,
        vision_model=vision_model or DEFAULT_VISION_MODEL,
        max_tokens=kwargs.pop("max_tokens", 700),
        **kwargs,
    )


def get_answer_Many(
    client: Any,
    examples: Iterable[Dict[str, Any]],
    method: str = "mammqa",
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        question = ex["question"]
        if verbose:
            print(f"\nExample {i + 1}: {question}")
        if method == "mammqa":
            res = get_answer_MM(
                client,
                question=question,
                text=ex.get("text", ""),
                tables=ex.get("table", ""),
                images=ex.get("images", []),
                text_model=text_model,
                vision_model=vision_model,
                qtype=ex.get("type"),
                verbose=verbose,
                **kwargs,
            )
            predicted = res.get("predicted_answer", "")
        elif method == "cot":
            raw = get_answer_cot(client, question, ex.get("text", ""), ex.get("table", ""), ex.get("images", []), text_model=text_model, vision_model=vision_model, **kwargs)
            res = {"Final Answer": raw}
            predicted = extract_answer_tag(raw)
        elif method == "zs":
            raw = get_answer_zs_no_data(client, question, model=text_model, **kwargs)
            res = {"Final Answer": raw}
            predicted = extract_answer_tag(raw)
        else:
            raise ValueError("method must be one of: mammqa, cot, zs")
        record = {
            "id": ex.get("id", str(i)),
            "question": question,
            "type": ex.get("type", ""),
            "method": method,
            "predicted_answer": predicted,
            "gold_answer": ex.get("answer", ""),
            **res,
        }
        records.append(record)
    return records
