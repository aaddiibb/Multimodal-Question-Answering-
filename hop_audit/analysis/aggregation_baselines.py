"""No-GPU test of the aggregation bottleneck: re-aggregate the agents' existing answers on the 300 traces.
Correctness: official EM, and 'content' = EM or the answer contains every gold item (SURFACE_FORM-type)."""
import json, sys, random
from collections import Counter
from pathlib import Path
sys.path.insert(0, "hop_audit")
from hop_audit import matching
from hop_audit.matching import mmqa_em_f1, split_prediction, mentions_all, is_abstaining_text, normalize
from hop_audit.gold import load_gold
from hop_audit.traces import read_mammqa_official
matching.load_aliases("hop_audit/aliases.json")
P = Path(sys.argv[1]); DEV = Path(sys.argv[2])
T = read_mammqa_official(P, "m"); G = load_gold(DEV, {t.qid for t in T})

def ok(ans, g):
    if not ans or is_abstaining_text(ans): return (0, 0)
    em = mmqa_em_f1(split_prediction(ans, len(g.answers)), g.answers)[0]
    return (int(em), int(em or mentions_all(g.answers, ans)))

def short(a): return a if a and len(normalize(a).split()) <= 12 and not is_abstaining_text(a) else ""

res = Counter(); per = []
for t in T:
    g = G[t.qid]; ag = {a.agent_id: a.answer for a in t.agents}
    s2 = [short(ag.get(f"S2.{m}", "")) for m in ("text", "table", "image")]
    allshort = [short(v) for v in ag.values()]
    def vote(cands, fallback):
        c = Counter(normalize(x) for x in cands if x)
        if not c: return fallback
        top, n = c.most_common(1)[0]
        if n < 2: return fallback
        return next(x for x in cands if x and normalize(x) == top)
    methods = {
        "MAMMQA final (as run)": t.final_answer,
        "S2 majority vote (>=2 agree) else final": vote(s2, t.final_answer),
        "all-agent majority vote else final": vote(allshort, t.final_answer),
        "S2.table answer else final": short(ag.get("S2.table", "")) or t.final_answer,
        "S2.text answer else final": short(ag.get("S2.text", "")) or t.final_answer,
    }
    row = {"qid": t.qid}
    for k, v in methods.items():
        em, ct = ok(v, g); res[(k, "em")] += em; res[(k, "content")] += ct; row[k] = (em, ct)
    oracle = max((ok(v, g) for v in list(ag.values()) + [t.final_answer]), key=lambda x: x[1])
    res[("ORACLE: any agent or final", "em")] += oracle[0]; res[("ORACLE: any agent or final", "content")] += oracle[1]
    row["oracle"] = oracle; per.append(row)

n = len(T)
names = list(dict.fromkeys(k for k, _ in res))
base = [r["MAMMQA final (as run)"] for r in per]
print(f"{'method':45s}  EM     content-correct   paired vs final (content): +gain / -loss")
for k in names:
    em, ct = res[(k, "em")] / n, res[(k, "content")] / n
    if k.startswith("ORACLE"):
        print(f"{k:45s}  {em:.3f}  {ct:.3f}"); continue
    gain = sum(r[k][1] > b[1] for r, b in zip(per, base)); loss = sum(r[k][1] < b[1] for r, b in zip(per, base))
    print(f"{k:45s}  {em:.3f}  {ct:.3f}            +{gain} / -{loss}")
Path("analysis/out_v02/aggregation_baselines.json").write_text(json.dumps({f"{k}|{m}": v for (k, m), v in res.items()}, indent=1))
