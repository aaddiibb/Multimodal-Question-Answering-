"""Gold hop structure from MultiModalQA dev metadata.

MMQA annotates every compositional question with `metadata.intermediate_answers`: a list of
sub-question answer sets, each item tagged with the modality and document it comes from.

* Compose(A, B): one set, the bridge entity produced by the inner question B.
* Compare(X, Y): two sets, the two things being compared (one is often named in the question).
* Intersect(X, Y): two candidate sets whose intersection is the final answer.
* TableQ with intermediate answers: the row-locator value (e.g. the season '2019-20').

Two refinements make hop recall meaningful:

* An item that already appears in the question (e.g. 'Nightwing' in a Compare question) is GIVEN,
  not latent. Surfacing it proves nothing, so only LATENT hops count toward bridge recall.
* For Intersect-style sets that overlap the final answer, only the overlapping items are required;
  otherwise every item of the set is required.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .matching import fmt_answer, mention_level, normalize


@dataclass
class HopItem:
    text: str
    modality: str
    doc_ids: List[str]
    in_question: bool


@dataclass
class Hop:
    index: int
    items: List[HopItem]
    required: List[int]                  # indices into items
    latent: bool                         # at least one required item is not named in the question
    modalities: List[str]

    @property
    def required_texts(self) -> List[str]:
        return [self.items[i].text for i in self.required]

    @property
    def latent_required_texts(self) -> List[str]:
        return [self.items[i].text for i in self.required if not self.items[i].in_question]


@dataclass
class GoldQuestion:
    qid: str
    question: str
    qtype: str
    top_type: str
    compositional: bool
    answers: List[str]
    answer_modalities: List[str]
    hops: List[Hop]
    support_modalities: List[str]
    pseudo_question: str = ""
    answer_in_question: bool = False
    uninformative_answer: bool = False
    answer_is_hop_item: bool = False     # e.g. Compare: the answer is one of the two compared items

    @property
    def latent_hops(self) -> List[Hop]:
        return [h for h in self.hops if h.latent]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_latent_hops"] = len(self.latent_hops)
        return d


def _item_docs(a: dict) -> List[str]:
    ids = [x.get("doc_id") for x in a.get("text_instances", []) or []]
    ids += [x.get("doc_id") for x in a.get("image_instances", []) or []]
    return [i for i in ids if i]


def build_gold(row: dict) -> GoldQuestion:
    md = row.get("metadata", {})
    q = row["question"]
    qtype = md.get("type", "")
    answers = [fmt_answer(a["answer"]) for a in row.get("answers", [])]
    ans_norm = {normalize(a) for a in answers}
    hops: List[Hop] = []
    for h_idx, hop_set in enumerate(md.get("intermediate_answers", []) or []):
        items = [HopItem(text=fmt_answer(a["answer"]), modality=a.get("modality", ""),
                         doc_ids=_item_docs(a),
                         in_question=mention_level(fmt_answer(a["answer"]), q, fuzzy=False) is not None)
                 for a in hop_set]
        overlap = [i for i, it in enumerate(items)
                   if normalize(it.text) in ans_norm
                   or any(mention_level(a, it.text, fuzzy=False) or mention_level(it.text, a, fuzzy=False)
                          for a in answers)]
        required = overlap if (overlap and len(items) > 1) else list(range(len(items)))
        latent = any(not items[i].in_question for i in required)
        hops.append(Hop(index=h_idx, items=items, required=required, latent=latent,
                        modalities=sorted({items[i].modality for i in required if items[i].modality})))
    support_mods = sorted({c.get("doc_part") for c in row.get("supporting_context", []) if c.get("doc_part")})
    hop_norm = {normalize(it.text) for h in hops for it in h.items}
    answer_is_hop_item = bool(answers) and all(normalize(a) in hop_norm for a in answers)
    return GoldQuestion(
        qid=row["qid"], question=q, qtype=qtype, top_type=qtype.split("(")[0],
        compositional=qtype.split("(")[0] in {"Compose", "Compare", "Intersect"},
        answers=answers,
        answer_modalities=sorted({a.get("modality", "") for a in row.get("answers", []) if a.get("modality")}),
        hops=hops, support_modalities=support_mods, pseudo_question=md.get("pseudo_language_question", ""),
        answer_in_question=bool(answers) and all(mention_level(a, q, fuzzy=False) for a in answers),
        uninformative_answer=bool(answers) and all(normalize(a) in {"yes", "no"} for a in answers),
        answer_is_hop_item=answer_is_hop_item,
    )


def load_gold(dev_path: Path, qids: Optional[Iterable[str]] = None) -> Dict[str, GoldQuestion]:
    want = set(qids) if qids is not None else None
    opener = gzip.open if str(dev_path).endswith(".gz") else open
    out: Dict[str, GoldQuestion] = {}
    with opener(dev_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if want is None or row["qid"] in want:
                out[row["qid"]] = build_gold(row)
    if want is not None:
        missing = want - set(out)
        if missing:
            raise KeyError(f"{len(missing)} qids not found in {dev_path}: {sorted(missing)[:3]}...")
    return out
