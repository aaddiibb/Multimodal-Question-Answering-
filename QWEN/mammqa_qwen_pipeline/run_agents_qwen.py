from __future__ import annotations

import argparse
import json
from pathlib import Path

from Dataloader import MultiModalQADataLoader
from Eval import evaluate_predictions, write_jsonl
from agents_qwen import get_answer_MM
from qwen_client import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL, make_qwen_client


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input/datasets/adibraian/multimodalqa-dataset")
    p.add_argument("--images-base-url", default=None)
    p.add_argument("--encode-images", action="store_true")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out", default="outputs/qwen_mammqa_predictions.jsonl")
    p.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    p.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    p.add_argument("--region", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--reduce-modalities-by-type", action="store_true")
    p.add_argument("--not-strict", action="store_true")
    args = p.parse_args()

    client = make_qwen_client(base_url=args.base_url, region=args.region)
    dl = MultiModalQADataLoader.from_root(args.root, images_base_url=args.images_base_url, encode_images=args.encode_images)
    records = []
    end = min(len(dl), args.start + args.n)
    for idx in range(args.start, end):
        ex = dl.get_agent_inputs(idx)
        print(f"\n[{idx}] {ex['type']} :: {ex['question']}")
        res = get_answer_MM(
            client,
            question=ex["question"],
            text=ex["text"],
            tables=ex["table"],
            images=ex["images"],
            text_model=args.text_model,
            vision_model=args.vision_model,
            qtype=ex["type"],
            strict_paper_mode=not args.not_strict,
            reduce_modalities_by_question_type=args.reduce_modalities_by_type,
            verbose=True,
        )
        records.append({
            "id": ex["id"],
            "question": ex["question"],
            "type": ex["type"],
            "method": "qwen_mammqa",
            "gold_answer": ex["answer"],
            **res,
        })
        print("Predicted:", res["predicted_answer"], "| Gold:", ex["answer"])
    write_jsonl(records, args.out)
    report = evaluate_predictions(records)
    report_path = str(Path(args.out).with_suffix(".eval.json"))
    Path(report_path).write_text(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), encoding="utf-8")
    print("\nSaved:", args.out)
    print("Report:", json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
