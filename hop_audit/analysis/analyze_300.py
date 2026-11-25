"""Pre-registered analysis of the 300-question hop-trace run (see PREREGISTRATION.md).

Primary: P = (COMPOSITION_FAILURE + AGGREGATION_LOSS + COMMITMENT_FAILURE) / |W|, W = wrong answers
excluding SURFACE_FORM; 95% bootstrap CI over questions (10,000 resamples, seed 0).
Robustness R1-R3 re-score with matching variants WITHOUT editing the frozen hop_audit code.
Usage: python analyze_300.py <hop_audit_dir> <mammqa_predictions.jsonl> <MMQA_dev.jsonl.gz> <out_dir>
"""
import json, random, sys, csv
from collections import Counter, defaultdict
from pathlib import Path

HOP, PRED, DEV, OUT = map(Path, sys.argv[1:5])
sys.path.insert(0, str(HOP))
from hop_audit import matching, scoring                       # noqa: E402
from hop_audit.gold import load_gold                          # noqa: E402
from hop_audit.traces import read_mammqa_official             # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
PRESENT = {"COMPOSITION_FAILURE", "AGGREGATION_LOSS", "COMMITMENT_FAILURE"}
B, SEED = 10000, 0


def run(variant: str):
    """Score all traces under a matching variant. Returns rows."""
    orig_mentions_all, orig_weak = matching.mentions_all, matching.is_weak_gold
    matching.ALIASES.clear()
    if variant != "R2_no_aliases":
        matching.load_aliases(HOP / "aliases.json")
    if variant == "R1_no_fuzzy":
        scoring.mentions_all = lambda golds, text, fuzzy=True: orig_mentions_all(golds, text, fuzzy=False)
    if variant == "R3_no_weak_guard":
        scoring.is_weak_gold = lambda g: False
    traces = read_mammqa_official(PRED, pipeline="mammqa300")
    gold = load_gold(DEV, {t.qid for t in traces})
    rows = [scoring.score_trace(t, gold[t.qid]) for t in traces]
    scoring.mentions_all, scoring.is_weak_gold = orig_mentions_all, orig_weak
    return rows, gold


def p_stat(rows):
    w = [r for r in rows if r["label"] not in ("CORRECT", "SURFACE_FORM")]
    return (sum(r["label"] in PRESENT for r in w) / len(w)) if w else float("nan"), len(w)


def boot(rows, stat, b=B, seed=SEED):
    rng = random.Random(seed)
    vals = []
    n = len(rows)
    for _ in range(b):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        v = stat(s)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def rate(num, den):
    return round(num / den, 4) if den else None


results = {}
rows, gold = run("primary")
P, nW = p_stat(rows)
lo, hi = boot(rows, lambda s: p_stat(s)[0])
verdict = "SUPPORTED" if lo > 0.5 else ("REFUTED" if hi < 0.5 else "INCONCLUSIVE")
labels = Counter(r["label"] for r in rows)
results["primary"] = {"n": len(rows), "EM": rate(sum(r["em"] for r in rows), len(rows)), "W": nW,
                      "present": sum(r["label"] in PRESENT for r in rows), "P": round(P, 4),
                      "CI95": [round(lo, 4), round(hi, 4)], "verdict_before_R4": verdict, "labels": dict(labels)}

# ---------------- secondary
by_type = defaultdict(Counter)
for r in rows:
    by_type[r["top_type"]][r["label"]] += 1
results["labels_by_type"] = {t: dict(c) for t, c in sorted(by_type.items())}
results["P_by_type"] = {}
for t in sorted(by_type):
    sub = [r for r in rows if r["top_type"] == t]
    p, w = p_stat(sub)
    results["P_by_type"][t] = {"P": round(p, 4) if w else None, "W": w}

# bridge recall by hop modality and stage
rec = defaultdict(lambda: Counter())
for r in rows:
    for h in r["hops"]:
        m = "+".join(h["modalities"])
        rec[m]["n"] += 1
        for s in (1, 2):
            rec[m][f"S{s}"] += bool(h["surfaced_by_stage"].get(s) or h["surfaced_by_stage"].get(str(s)))
        rec[m]["any"] += h["first_stage_surfaced"] is not None
results["bridge_recall_by_modality"] = {m: {"n": c["n"], "S1": rate(c["S1"], c["n"]), "S2": rate(c["S2"], c["n"]),
                                            "any": rate(c["any"], c["n"])} for m, c in sorted(rec.items())}


def strict(s):
    car = drp = 0
    for r in s:
        for h in r["hops"]:
            for c in h["channels"]:
                car += c["outcome"] == "carried"; drp += c["outcome"] == "dropped"
    return drp / (car + drp) if (car + drp) else float("nan")


lo_s, hi_s = boot(rows, strict, b=2000)
chan = defaultdict(Counter)
for r in rows:
    for h in r["hops"]:
        for c in h["channels"]:
            chan[f"{c['from']}->{c['to']}"][c["outcome"]] += 1
results["strict_handoff"] = {"drop_rate": round(strict(rows), 4), "CI95": [round(lo_s, 4), round(hi_s, 4)],
                             "by_channel": {k: dict(v) for k, v in sorted(chan.items())}}
comp = [r for r in rows if r["label"] == "COMPOSITION_FAILURE"]
results["composition_failures"] = {"n": len(comp), "final_abstained": sum(r["final_abstain"] for r in comp),
                                   "abstain_rate": rate(sum(r["final_abstain"] for r in comp), len(comp))}
em_t = defaultdict(list)
for r in rows:
    em_t[r["top_type"]].append(r["em"])
results["EM_by_type"] = {t: [rate(sum(v), len(v)), len(v)] for t, v in sorted(em_t.items())}

# ---------------- robustness R1-R3
results["robustness"] = {}
for v in ["R1_no_fuzzy", "R2_no_aliases", "R3_no_weak_guard"]:
    rv, _ = run(v)
    p, w = p_stat(rv)
    l, h = boot(rv, lambda s: p_stat(s)[0], b=2000)
    changed = sum(a["label"] != b["label"] for a, b in zip(rows, rv))
    results["robustness"][v] = {"P": round(p, 4), "W": w, "CI95": [round(l, 4), round(h, 4)],
                                "labels_changed_vs_primary": changed, "labels": dict(Counter(r["label"] for r in rv))}

(OUT / "analysis_300.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
with open(OUT / "labels_300.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["qid", "top_type", "type", "label", "em", "f1", "final_abstain", "prediction", "gold", "hop_statuses"])
    for r in rows:
        w.writerow([r["qid"], r["top_type"], r["type"], r["label"], r["em"], r["f1"], r["final_abstain"], r["prediction"],
                    " | ".join(r["gold"]), ";".join(r["hop_statuses"])])
print(json.dumps(results, indent=1))
