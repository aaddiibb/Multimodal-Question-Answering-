"""String matching used by the hop audit.

Two different questions are asked of model outputs, and they use different rules:

* Final-answer correctness uses the MultiModalQA official metric (DROP-style bag EM/F1 with
  Hungarian alignment), so hop-audit numbers line up with the leaderboard metric.
* "Did this agent surface gold item X?" (mention) uses normalized whole-token containment,
  with a conservative fuzzy fallback for long names. This is what bridge recall is built on.

Everything here is pure standard library, so the audit runs on any Python >= 3.8.
"""
from __future__ import annotations

import itertools
import re
import string
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_PUNC = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—"}
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_DISAMBIG = re.compile(r"\s*\([^()]*\)\s*$")

ABSTAIN_VALUES = {
    "", "insufficient", "abstain", "unknown", "none", "n/a", "na", "not enough information",
    "cannot be determined", "cannot determine", "unable to determine", "no answer",
}
_ABSTAIN_RE = re.compile(
    r"^(final\s*:\s*)?(insufficient|abstain|unknown|none|not enough information|cannot be determined|"
    r"can ?not (be )?determined?|unable to determine|no (relevant )?information)\b", re.I)
UNINFORMATIVE_GOLD = {"yes", "no", "true", "false"}


# ----------------------------------------------------------------------------- normalisation
def fmt_answer(value) -> str:
    """MMQA stores some golds as floats (1976.0). Render integral floats as integers."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _is_number(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def _norm_tok(t: str) -> str:
    t = t.lower()
    if not _is_number(t):
        t = "".join(ch for ch in t if ch not in _PUNC)
    t = _ARTICLES.sub(" ", t)
    if _is_number(t):
        t = str(float(t))
    return " ".join(t.split())


def normalize(text) -> str:
    """MMQA / DROP answer normalisation (lowercase, drop punctuation and articles, canonical numbers),
    plus accent folding so 'Dženis' matches 'Dzenis'."""
    text = _strip_accents(fmt_answer(text))
    # v0.2: en/em dashes separate tokens like hyphens ('3–3' == '3-3'); v0.1 merged them ('33').
    parts = [_norm_tok(tok) for tok in re.split(r"\s|[-–—]|/", text)]
    return " ".join(p for p in parts if p.strip()).strip()


_NUM_WORDS = {w: str(float(i)) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}

# Human-reviewed aliases (normalised gold -> list of accepted surface forms). Used ONLY for mention /
# commitment checks, never for the official EM. Loaded from aliases.json by load_aliases().
ALIASES: Dict[str, List[str]] = {}


def load_aliases(path) -> int:
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return 0
    data = json.loads(p.read_text(encoding="utf-8"))
    ALIASES.clear()
    for gold, alts in data.get("aliases", {}).items():
        ALIASES[normalize(gold)] = list(alts)
    return len(ALIASES)


def normalize_mention(text) -> str:
    """normalize() plus number words -> digits ('eight' == '8'), for mention checks only."""
    return " ".join(_NUM_WORDS.get(t, t) for t in normalize(text).split())


def variants(gold: str) -> List[str]:
    """Surface variants accepted for a gold item: as-is, without a trailing Wikipedia
    disambiguator ('Mask (1985 film)' -> 'Mask'), and any reviewed aliases."""
    out = [gold]
    bare = _DISAMBIG.sub("", gold).strip()
    if bare and bare != gold:
        out.append(bare)
    out += ALIASES.get(normalize(gold), [])
    return out


_ABSTAIN_SENTENCE = re.compile(
    r"\b(insufficient|not enough information|(not|impossible to|unable to|cannot|can ?not) (be )?(determine|determined|identify|identified|answer)|"
    r"does not (provide|contain|mention|include)|do not (provide|contain|mention|include)|"
    r"no (specific |relevant )?information|not (provided|mentioned|specified|available)|unanswerable)\b", re.I)


def is_abstaining_text(answer: Optional[str]) -> bool:
    """True for answer fields that are abstentions, including full sentences such as
    'There is insufficient information to determine ...'."""
    return is_abstain(answer) or bool(_ABSTAIN_SENTENCE.search(answer or ""))


def is_weak_gold(gold) -> bool:
    """Golds that turn up by accident in long outputs: yes/no, numbers, number words, <= 3 characters,
    and single lowercase common nouns ('horses', 'blue'). Single capitalised names ('Essendon') stay strong."""
    toks = normalize_mention(gold).split()
    if len(toks) != 1:
        return False
    return (_is_number(toks[0]) or len(toks[0]) <= 3 or toks[0] in UNINFORMATIVE_GOLD
            or fmt_answer(gold).strip()[:1].islower())


def n_tokens(text: Optional[str]) -> int:
    return len(normalize(text or "").split())


# ----------------------------------------------------------------------------- mention matching
def mention_level(gold, text: Optional[str], fuzzy: bool = True) -> Optional[str]:
    """Return 'exact', 'fuzzy' or None: does `text` mention the gold item?

    exact: the normalised gold occurs in the normalised text as whole tokens.
    fuzzy: only for golds of >= 3 tokens; some window of the text of the same length shares
           >= 80% of the gold's tokens (catches 'Tell Me You Love Me, Junie Moon' vs the full title).
    """
    if text is None:
        return None
    t = normalize_mention(text)
    if not t:
        return None
    padded = f" {t} "
    for v in variants(fmt_answer(gold)):
        g = normalize_mention(v)
        if g and f" {g} " in padded:
            return "exact"
    if not fuzzy:
        return None
    t_toks = t.split()
    for v in variants(fmt_answer(gold)):
        g_toks = normalize_mention(v).split()
        n = len(g_toks)
        if n < 3 or len(t_toks) < n:
            continue
        need = set(g_toks)
        for i in range(0, len(t_toks) - n + 1):
            window = set(t_toks[i:i + n + 1])
            if len(need & window) / len(need) >= 0.8:
                return "fuzzy"
    return None


def mentions_all(golds: Sequence, text: Optional[str], fuzzy: bool = True) -> bool:
    return bool(golds) and all(mention_level(g, text, fuzzy) for g in golds)


def is_abstain(answer: Optional[str]) -> bool:
    a = (answer or "").strip().strip(".").strip()
    return a.lower() in ABSTAIN_VALUES or bool(_ABSTAIN_RE.match(a))


def is_uninformative(gold) -> bool:
    """Golds whose mere mention says nothing (yes/no)."""
    return normalize(gold) in UNINFORMATIVE_GOLD


# ----------------------------------------------------------------------------- official MMQA EM/F1
def _bags(spans: Sequence[str]) -> Tuple[List[str], List[set]]:
    norm = [normalize(s) for s in spans]
    return norm, [set(n.split()) for n in norm]


def _f1(pred_bag: set, gold_bag: set) -> float:
    inter = len(gold_bag & pred_bag)
    p = 1.0 if not pred_bag else inter / len(pred_bag)
    r = 1.0 if not gold_bag else inter / len(gold_bag)
    return 0.0 if p == 0 and r == 0 else 2 * p * r / (p + r)


def _numbers_match(gold_bag: set, pred_bag: set) -> bool:
    g = {w for w in gold_bag if _is_number(w)}
    return (not g) or bool(g & {w for w in pred_bag if _is_number(w)})


def _max_assignment(scores: List[List[float]]) -> List[Tuple[int, int]]:
    """Maximum-weight assignment. scipy if available, exact brute force for small cases, else greedy."""
    if not scores or not scores[0]:
        return []
    try:  # pragma: no cover - exercised only when scipy is installed
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        r, c = linear_sum_assignment(-np.array(scores))
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        pass
    n_r, n_c = len(scores), len(scores[0])
    if min(n_r, n_c) <= 7 and max(n_r, n_c) <= 9:
        best, best_pairs = -1.0, []
        if n_r <= n_c:
            for perm in itertools.permutations(range(n_c), n_r):
                s = sum(scores[i][perm[i]] for i in range(n_r))
                if s > best:
                    best, best_pairs = s, [(i, perm[i]) for i in range(n_r)]
        else:
            for perm in itertools.permutations(range(n_r), n_c):
                s = sum(scores[perm[j]][j] for j in range(n_c))
                if s > best:
                    best, best_pairs = s, [(perm[j], j) for j in range(n_c)]
        return best_pairs
    pairs, used_r, used_c = [], set(), set()
    cells = sorted(((scores[i][j], i, j) for i in range(n_r) for j in range(n_c)), reverse=True)
    for _, i, j in cells:
        if i not in used_r and j not in used_c:
            pairs.append((i, j)); used_r.add(i); used_c.add(j)
    return pairs


def mmqa_em_f1(predicted: Sequence[str], gold: Sequence[str]) -> Tuple[float, float]:
    """MultiModalQA official EM/F1 (DROP-style bags + optimal alignment)."""
    predicted = [fmt_answer(p) for p in predicted] or [""]
    gold = [fmt_answer(g) for g in gold]
    pn, pb = _bags(predicted)
    gn, gb = _bags(gold)
    em = 1.0 if (set(pn) == set(gn) and len(pn) == len(gn)) else 0.0
    scores = [[(_f1(p, g) if _numbers_match(g, p) else 0.0) for p in pb] for g in gb]
    best = [0.0] * max(len(gb), len(pb))
    for i, j in _max_assignment(scores):
        best[i] = max(best[i], scores[i][j])
    return em, round(sum(best) / len(best), 4)


def split_prediction(pred: str, n_gold: int) -> List[str]:
    """Split a free-text prediction into answer items. '|' and ';' always separate items; commas and
    ' and ' only separate when the gold itself is a list (so 'Washington, D.C.' stays one item)."""
    pred = (pred or "").strip()
    parts = re.split(r"\s*\|\s*|\s*;\s*", pred)
    if n_gold > 1:
        parts = [q for p in parts for q in re.split(r"\s*,\s*|\s+and\s+", p)]
    parts = [p.strip().strip(".") for p in parts if p.strip()]
    return parts or [pred]


def iter_nonempty(xs: Iterable[Optional[str]]) -> List[str]:
    return [x for x in xs if x]
