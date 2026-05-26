"""Evaluation utilities for MAMMQA/Qwen predictions."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_answer_tag(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else str(text or "").strip()


def _strip_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = _strip_articles(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: Any, gold: Any) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


_NLP = None

def _lemma_tokens(text: Any) -> List[str]:
    global _NLP
    if _NLP is None:
        try:
            import spacy  # type: ignore
            try:
                _NLP = spacy.load("en_core_web_sm")
            except Exception:
                _NLP = spacy.blank("en")
        except Exception:
            _NLP = False
    norm = normalize_answer(text)
    if not norm:
        return []
    if _NLP is False:
        return norm.split()
    doc = _NLP(norm)  # type: ignore[operator]
    toks = []
    for tok in doc:
        lemma = getattr(tok, "lemma_", "") or tok.text
        lemma = normalize_answer(lemma)
        if lemma:
            toks.append(lemma)
    return toks or norm.split()


def lemma_match(prediction: Any, gold: Any) -> bool:
    p = _lemma_tokens(prediction)
    g = _lemma_tokens(gold)
    return bool(p and g and " ".join(p) == " ".join(g))


def _gold_list(gold: Any) -> List[str]:
    if gold is None:
        return [""]
    if isinstance(gold, list):
        return [str(x) for x in gold]
    if isinstance(gold, str) and " | " in gold:
        return [x.strip() for x in gold.split(" | ") if x.strip()]
    return [str(gold)]


def evaluate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    pred = record.get("predicted_answer") or extract_answer_tag(record.get("Final Answer", ""))
    golds = _gold_list(record.get("gold_answer") or record.get("answer") or record.get("true_answer"))
    em = max(exact_match(pred, g) for g in golds)
    f1 = max(token_f1(pred, g) for g in golds)
    lem = max(lemma_match(pred, g) for g in golds)
    out = dict(record)
    out.update({"predicted_answer": pred, "exact_match": bool(em), "f1": float(f1), "lemma_match": bool(lem)})
    return out


def evaluate_predictions(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [evaluate_record(r) for r in records]
    n = len(evaluated) or 1
    return {
        "n": len(evaluated),
        "exact_match": sum(1 for r in evaluated if r["exact_match"]) / n,
        "token_f1": sum(float(r["f1"]) for r in evaluated) / n,
        "lemma_match_accuracy": sum(1 for r in evaluated if r["lemma_match"]) / n,
        "records": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Prediction JSONL containing predicted_answer and gold_answer fields.")
    parser.add_argument("--out", default="eval_report.json")
    args = parser.parse_args()
    report = evaluate_predictions(read_jsonl(args.pred))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
