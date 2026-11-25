"""Run the hop-level audit.

Default (no arguments): audits the traces already in this project folder and writes to
hop_audit/outputs/:
  * cv_pilot_baseline   cv_mammqa_16 shared Stage I/II + MAMMQA-style aggregator
  * cv_pilot_cv         cv_mammqa_16 shared Stage I/II + claim-verification aggregator
  * hm_rag_oracle       HM-RAG + MAMMQA smoke run, oracle-retrieval arm
  * mammqa_full_outputs released-prompt MAMMQA run (20 questions, images unavailable)

Custom traces:
  python run_hop_audit.py --source NAME=READER:PATH[,FINAL_PATH]
  e.g. --source my_run=mammqa_official:path/to/mammqa_predictions.jsonl

Readers: cv_pilot, mammqa_official, hm_oracle (see hop_audit/traces.py to add one).

Gold statistics for the whole dev set:  python run_hop_audit.py --dev-stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hop_audit.gold import load_gold                    # noqa: E402
from hop_audit.matching import load_aliases              # noqa: E402
from hop_audit.report import gold_stats, write_pipeline, write_report  # noqa: E402
from hop_audit.scoring import score_trace, summarize    # noqa: E402
from hop_audit.traces import READERS                    # noqa: E402

ROOT = HERE.parent
DEV = ROOT / "MAMMQA-main" / "Codes" / "multimodalqa-dataset" / "MMQA_dev.jsonl.gz"
CV = ROOT / "cv_mammqa_16" / "predictions"
HM = (ROOT / "New folder" / "HM-RAG + MAMMQA 16-question smoke experiment" / "hm_mammqa" /
      "qa_smoke16_full_pipeline_seed42" / "predictions" / "MAMMQA_ORACLE.jsonl")
MF = ROOT / "mammqa_full_outputs" / "mammqa_predictions.jsonl"

DEFAULT_SOURCES = {
    "cv_pilot_baseline": ("cv_pilot", [CV / "predictions_shared.jsonl", CV / "predictions_mammqa.jsonl"]),
    "cv_pilot_cv": ("cv_pilot", [CV / "predictions_shared.jsonl", CV / "predictions_cv_mammqa.jsonl"]),
    "hm_rag_oracle": ("hm_oracle", [HM]),
    "mammqa_full_outputs": ("mammqa_official", [MF]),
}
NOTES = {
    "cv_pilot_baseline": "Kaggle CV-MAMMQA pilot (debate-mammqa v358107244), Qwen2.5-VL-7B 4-bit, gold supporting docs only, "
                         "image titles given as text. Short output caps (Stage I 112, Stage II 128 tokens). Final = baseline aggregator.",
    "cv_pilot_cv": "Same Stage I/II outputs as cv_pilot_baseline; final = claim-verification aggregator.",
    "hm_rag_oracle": "HM-RAG + MAMMQA smoke run, MAMMQA_ORACLE arm (gold docs as retrieval). Cross-modal agents' raw inputs are "
                     "not logged, so only stage-level hand-off loss is computed.",
    "mammqa_full_outputs": "Released-prompt MAMMQA run, 20 questions, image modality unavailable (no images given), so image hops "
                           "are NO_EVIDENCE by construction.",
}

KNOWN_SHA256 = {CV / "predictions_shared.jsonl": "f17b8d46524f559453c2213a28c87c92faadae84ac254e643fcfc1c97f098969",
                CV / "predictions_mammqa.jsonl": "f5599a5ba2d2b45b68693ff3c4061465f8191e5b69e8162650e4dc9ea5479108",
                CV / "predictions_cv_mammqa.jsonl": "1b9e27a271ab1b64ffc9650a52550dc008fea1f1c6a6aad03f7e156371f69352"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    """Path relative to the project folder when possible (keeps manifests portable across machines)."""
    try:
        return Path(p).absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError:
        return str(p)


def parse_source(spec: str):
    name, rest = spec.split("=", 1)
    reader, paths = rest.split(":", 1) if rest.split(":", 1)[0] in READERS else (None, rest)
    if reader is None:
        raise SystemExit(f"Unknown reader in {spec!r}; choose from {sorted(READERS)}")
    return name, (reader, [Path(p) for p in paths.split(",")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, default=DEV, help="MMQA_dev.jsonl(.gz)")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--source", action="append", default=[], help="NAME=READER:PATH[,FINAL_PATH]")
    ap.add_argument("--dev-stats", action="store_true", help="also compute gold hop statistics on all of dev")
    ap.add_argument("--aliases", type=Path, default=HERE / "aliases.json",
                    help="human-reviewed aliases for mention checks (never used for EM)")
    ap.add_argument("--no-aliases", action="store_true")
    args = ap.parse_args(argv)
    n_alias = 0 if args.no_aliases else load_aliases(args.aliases)
    print(f"[aliases] {n_alias} reviewed alias entries loaded")

    sources = dict(parse_source(s) for s in args.source) if args.source else DEFAULT_SOURCES
    manifest = {"dev": rel(args.dev), "dev_sha256": sha256(args.dev) if args.dev.exists() else None, "inputs": {}}
    traces = {}
    for name, (reader, paths) in sources.items():
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[skip] {name}: missing {missing[0]}")
            continue
        for p in paths:
            h = sha256(p)
            manifest["inputs"][rel(p)] = h
            if p in KNOWN_SHA256 and KNOWN_SHA256[p] != h:
                print(f"[warn] {p.name} differs from the frozen Kaggle output (sha256 {h[:16]})")
        traces[name] = READERS[reader](*paths, pipeline=name)
        print(f"[load] {name}: {len(traces[name])} traces via {reader}")

    qids = {t.qid for ts in traces.values() for t in ts}
    gold_all = load_gold(args.dev) if args.dev_stats else None
    gold = {q: gold_all[q] for q in qids} if gold_all else load_gold(args.dev, qids)

    results = {}
    for name, ts in traces.items():
        rows = [score_trace(t, gold[t.qid]) for t in ts]
        summary = summarize(rows)
        write_pipeline(args.out / name, rows, summary)
        results[name] = {"rows": rows, "summary": summary}
        lab = {k: v for k, v in summary["labels"].items() if v}
        print(f"[done] {name}: EM={summary['EM']} labels={lab}")

    dev_stats = gold_stats(gold_all) if gold_all else {}
    write_report(args.out / "report.md", results, dev_stats, NOTES)
    manifest["dev_stats"] = dev_stats
    manifest["aliases"] = {"path": None if args.no_aliases else rel(args.aliases), "entries": n_alias}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[report] {rel(args.out / 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
