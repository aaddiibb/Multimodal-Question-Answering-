from __future__ import annotations

import argparse
import json
from pathlib import Path

from Dataloader import MultiModalQADataLoader
from Eval import evaluate_predictions, write_jsonl
from agents_qwen import extract_answer_tag, get_answer_cot, get_answer_zs_no_data
from qwen_client import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL, make_qwen_client


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input/datasets/adibraian/multimodalqa-dataset")
    p.add_argument("--images-base-url", default=None)
    p.add_argument("--encode-images", action="store_true")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--method", choices=["zs", "cot"], default="cot")
    p.add_argument("--out", default="outputs/qwen_baseline_predictions.jsonl")
    p.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    p.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    p.add_argument("--region", default=None)
    p.add_argument("--base-url", default=None)
    args = p.parse_args()

    client = make_qwen_client(base_url=args.base_url, region=args.region)
    dl = MultiModalQADataLoader.from_root(args.root, images_base_url=args.images_base_url, encode_images=args.encode_images)
    records = []
    end = min(len(dl), args.start + args.n)
    for idx in range(args.start, end):
        ex = dl.get_agent_inputs(idx)
        if args.method == "zs":
            raw = get_answer_zs_no_data(client, ex["question"], model=args.text_model)
        else:
            raw = get_answer_cot(client, ex["question"], ex["text"], ex["table"], ex["images"], text_model=args.text_model, vision_model=args.vision_model)
        pred = extract_answer_tag(raw)
        print(f"[{idx}] pred={pred!r} gold={ex['answer']!r}")
        records.append({
            "id": ex["id"], "question": ex["question"], "type": ex["type"],
            "method": "qwen_" + args.method, "predicted_answer": pred,
            "gold_answer": ex["answer"], "Final Answer": raw,
        })
    write_jsonl(records, args.out)
    report = evaluate_predictions(records)
    report_path = str(Path(args.out).with_suffix(".eval.json"))
    Path(report_path).write_text(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), encoding="utf-8")
    print("Saved:", args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
