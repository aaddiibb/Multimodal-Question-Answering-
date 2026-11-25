"""Frozen sample for the hop-trace run: 300 MMQA dev questions with >= 1 latent gold hop,
proportional to question type, excluding questions inspected in earlier pilots, in random order."""
import sys, json, gzip, random, hashlib, collections
from pathlib import Path
sys.path.insert(0, sys.argv[1])                     # hop_audit package dir
from hop_audit.gold import build_gold
dev = [json.loads(l) for l in gzip.open(sys.argv[2], "rt")]
excluded = set(json.load(open(sys.argv[3])))
N, SEED = 300, 20261030
pool = collections.defaultdict(list)
for r in dev:
    g = build_gold(r)
    if g.latent_hops and r["qid"] not in excluded:
        pool[g.top_type].append(r["qid"])
total = sum(len(v) for v in pool.values())
types = sorted(pool)
raw = {t: N * len(pool[t]) / total for t in types}
alloc = {t: int(raw[t]) for t in types}
for t in sorted(types, key=lambda t: -(raw[t] - int(raw[t])))[: N - sum(alloc.values())]:
    alloc[t] += 1
rng = random.Random(SEED)
chosen = []
for t in types:
    chosen += [(q, t) for q in rng.sample(sorted(pool[t]), alloc[t])]
rng.shuffle(chosen)                                  # processing order: any prefix is a random subsample
sel = {"seed": SEED, "n": N, "criterion": "MMQA dev, >=1 latent gold hop (hop_audit.gold), excluding 52 previously inspected qids",
       "pool_sizes": {t: len(pool[t]) for t in types}, "allocation": alloc,
       "questions": [{"qid": q, "top_type": t} for q, t in chosen]}
blob = json.dumps(sel["questions"], sort_keys=True)
sel["sha256_questions"] = hashlib.sha256(blob.encode()).hexdigest()
Path(sys.argv[4]).write_text(json.dumps(sel, indent=1), encoding="utf-8")
print(alloc, sel["pool_sizes"], sel["sha256_questions"][:16])
