from __future__ import annotations

import argparse
import json

from Dataloader import MultiModalQADataLoader
from Eval import evaluate_predictions, write_jsonl
from qwen_client import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL, make_qwen_client
from tot_dfs_qwen import run_dfs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input/datasets/adibraian/multimodalqa-dataset")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--images-base-url", default=None)
    p.add_argument("--encode-images", action="store_true")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--beam-threshold", type=float, default=0.45)
    p.add_argument("--out", default="outputs/qwen_tot_prediction.jsonl")
    p.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    p.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    p.add_argument("--region", default=None)
    p.add_argument("--base-url", default=None)
    args = p.parse_args()

    client = make_qwen_client(base_url=args.base_url, region=args.region)
    dl = MultiModalQADataLoader.from_root(args.root, images_base_url=args.images_base_url, encode_images=args.encode_images)
    ex = dl.get_agent_inputs(args.idx)
    result = run_dfs(
        client,
        question=ex["question"],
        text=ex["text"],
        table=ex["table"],
        images=ex["images"],
        k=args.k,
        max_depth=args.max_depth,
        beam_threshold=args.beam_threshold,
        text_model=args.text_model,
        vision_model=args.vision_model,
        verbose=True,
    )
    record = {
        "id": ex["id"], "question": ex["question"], "type": ex["type"],
        "method": "qwen_tot", "predicted_answer": result["predicted_answer"],
        "gold_answer": ex["answer"], "Final Answer": result["final_raw"], "tot": result,
    }
    write_jsonl([record], args.out)
    print(json.dumps(evaluate_predictions([record]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
