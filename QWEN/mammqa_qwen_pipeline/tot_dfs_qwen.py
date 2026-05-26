"""DFS driver for Qwen Tree-of-Thoughts experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from agents_qwen import extract_answer_tag
from qwen_client import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL
from treeofthoughts_qwen import expand_thought, final_answer_from_path, generate_initial_thoughts, score_thought


def run_dfs(
    client: Any,
    question: str,
    text: str = "",
    table: str = "",
    images: Optional[Sequence[Any]] = None,
    initial_thoughts: Optional[List[Dict[str, Any]]] = None,
    k: int = 3,
    beam_threshold: float = 0.45,
    max_depth: int = 3,
    text_model: str = DEFAULT_TEXT_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a small DFS Tree-of-Thoughts search with pruning."""
    if initial_thoughts is None:
        initial_thoughts = generate_initial_thoughts(
            client, question, text=text, table=table, images=images, k=k, text_model=text_model, vision_model=vision_model
        )

    best: Dict[str, Any] = {"score": -1.0, "path": [], "final_raw": "", "predicted_answer": ""}

    def consider_final(path: List[Dict[str, Any]], score: float) -> None:
        nonlocal best
        raw = final_answer_from_path(
            client, question, path, text=text, table=table, images=images, text_model=text_model, vision_model=vision_model
        )
        if score > best["score"]:
            best = {"score": score, "path": path, "final_raw": raw, "predicted_answer": extract_answer_tag(raw)}

    def dfs(path: List[Dict[str, Any]], depth: int) -> None:
        if not path:
            return
        scored = score_thought(
            client, question, path[-1], text=text, table=table, images=images, text_model=text_model, vision_model=vision_model
        )
        score = float(scored.get("score", 0.0))
        path[-1]["score"] = score
        path[-1]["score_rationale"] = scored.get("rationale", "")
        if verbose:
            print(f"depth={depth} score={score:.2f} thought={str(path[-1])[:120]}")
        if score < beam_threshold:
            return
        if depth >= max_depth:
            consider_final(path, score)
            return
        children = expand_thought(
            client, question, path, text=text, table=table, images=images, k=k, text_model=text_model, vision_model=vision_model
        )
        # Score children before recursing, keep top k.
        scored_children = []
        for child in children:
            child_score = score_thought(
                client, question, child, text=text, table=table, images=images, text_model=text_model, vision_model=vision_model
            )
            child["score"] = float(child_score.get("score", 0.0))
            child["score_rationale"] = child_score.get("rationale", "")
            scored_children.append(child)
        scored_children.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        for child in scored_children[:k]:
            dfs(path + [child], depth + 1)
        # Also allow stopping early if the current path is already good.
        consider_final(path, score)

    roots = []
    for thought in initial_thoughts:
        roots.append(thought if isinstance(thought, dict) else {"thought": str(thought)})
    # Prioritize better initial thoughts.
    for root in roots:
        root_score = score_thought(client, question, root, text=text, table=table, images=images, text_model=text_model, vision_model=vision_model)
        root["score"] = float(root_score.get("score", 0.0))
        root["score_rationale"] = root_score.get("rationale", "")
    roots.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    for root in roots[:k]:
        dfs([root], 1)
    return best
