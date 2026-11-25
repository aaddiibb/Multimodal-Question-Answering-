"""R4 sample (seed 7, 60 non-CORRECT questions) and blind case dumps (no automatic label shown)."""
import json, random, re, sys
from pathlib import Path
sys.path.insert(0, "hop_audit")
from hop_audit import matching
from hop_audit.gold import load_gold
from hop_audit.traces import read_mammqa_official
matching.load_aliases("hop_audit/aliases.json")
PRED = sys.argv[1]
rows = {r["qid"]: r for r in (json.loads(l) for l in open("analysis/out/../../out300/mammqa300/per_question.jsonl"))}
non_correct = sorted(q for q, r in rows.items() if r["label"] != "CORRECT")
sample = random.Random(7).sample(non_correct, 60)
Path("analysis/out/r4_sample.json").write_text(json.dumps(sample))
traces = {t.qid: t for t in read_mammqa_official(Path(PRED), "m")}
gold = load_gold(Path("MAMMQA-main/Codes/multimodalqa-dataset/MMQA_dev.jsonl.gz"), set(sample))
def clean(s):
    s = re.sub(r"</?(modality|insights|answer|subanswer)>", " ", s or "")
    return " ".join(s.split())
def mark(s, golds, hops):
    for g in sorted(golds, key=len, reverse=True):
        s = re.sub(re.escape(g), lambda m: f"⟦{m.group(0)}⟧", s, flags=re.I)
    for h in sorted(hops, key=len, reverse=True):
        s = re.sub(re.escape(h), lambda m: f"«{m.group(0)}»", s, flags=re.I)
    return s
out = []
for i, q in enumerate(sample, 1):
    t, g = traces[q], gold[q]
    hops = [it for h in g.latent_hops for it in h.latent_required_texts]
    out.append(f"\n=== CASE {i} | {q[:8]} | {g.qtype}\nQ: {g.question}\nGOLD: {g.answers}\nLATENT HOPS: " +
               "; ".join(f"{h.latent_required_texts} [{','.join(h.modalities)}]" for h in g.latent_hops))
    for a in t.agents:
        txt = clean(a.text)
        out.append(f"  {a.agent_id}: {mark(txt[:900], g.answers, hops)}{' …' if len(txt) > 900 else ''}")
    out.append(f"  FINAL: {mark(clean(t.final_text)[-400:], g.answers, hops)}  || PRED={t.final_answer!r}")
Path("analysis/out/r4_cases.txt").write_text("\n".join(out), encoding="utf-8")
print(len(sample), len("\n".join(out)))
