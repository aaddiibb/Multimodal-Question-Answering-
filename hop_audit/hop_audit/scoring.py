"""Hop-level scoring and automatic failure attribution.

For every question the scorer asks, per latent gold hop and per agent:
  available  - could this agent see the hop's evidence as raw input?
  surfaced   - did this agent's output mention every required item of the hop?
and per channel (upstream agent U -> downstream agent D, where D can reach the hop ONLY via U):
  carried / dropped - strict hand-off outcome.

Each wrong final answer then gets ONE label: the earliest point at which the needed information
was lost, checked in this order:

  SURFACE_FORM         final answer contains every gold item, but official EM is 0 (formatting,
                       extra items, disambiguators). Content right, metric strict.
  AGGREGATION_LOSS     some intermediate agent committed to the gold answer; the final did not.
  COMMITMENT_FAILURE   the gold answer appears in some agent's reasoning, but no agent committed
                       to it (e.g. 'James Taylor' named in an insight, answer INSUFFICIENT).
  NO_EVIDENCE          a needed hop (or the answer) lives in a modality absent from the context.
  EXTRACTION_FAILURE   a needed hop was available but no agent ever surfaced it
                       (for questions without latent hops: the answer was never surfaced).
  HANDOFF_LOSS         a latent hop was surfaced at an earlier stage but no agent at the last
                       intermediate stage carries it.
  COMPOSITION_FAILURE  every latent hop reached the last intermediate stage, yet no agent
                       produced the gold answer.

The commitment and aggregation checks come first on purpose: if the gold answer was already in
hand somewhere in the pipeline, the hops were not the bottleneck.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .gold import GoldQuestion, Hop
from .matching import (is_abstaining_text, is_weak_gold, mention_level, mentions_all, mmqa_em_f1, n_tokens,
                       normalize, split_prediction)
from .traces import AgentOutput, Trace

LABEL_ORDER = ["CORRECT", "SURFACE_FORM", "AGGREGATION_LOSS", "COMMITMENT_FAILURE", "NO_EVIDENCE",
               "EXTRACTION_FAILURE", "HANDOFF_LOSS", "COMPOSITION_FAILURE"]
HOP_STATUS_ORDER = ["NO_EVIDENCE", "NOT_SURFACED", "LOST_IN_HANDOFF", "REACHED"]


def _raw_available(agent: AgentOutput, hop: Hop, ctx: Optional[List[str]]) -> Optional[bool]:
    if agent.raw_inputs is None:
        return None
    mods = set(hop.modalities)
    if ctx is not None:
        mods &= set(ctx)
    return bool(mods & set(agent.raw_inputs))


def score_hop(hop: Hop, trace: Trace) -> dict:
    texts = hop.latent_required_texts
    surfaced = {a.agent_id: mentions_all(texts, a.text) for a in trace.agents}
    match_kind = {a.agent_id: [mention_level(t, a.text) for t in texts] for a in trace.agents}
    available = {a.agent_id: _raw_available(a, hop, trace.context_modalities) for a in trace.agents}
    stages = sorted({a.stage for a in trace.agents})
    synth = sorted({a.stage for a in trace.agents if not a.decision})
    last = synth[-1] if synth else 0          # hops must reach the last SYNTHESIS stage, not the verifier
    by_stage = {s: any(surfaced[a.agent_id] for a in trace.agents if a.stage == s) for s in stages}
    in_context = (None if trace.context_modalities is None
                  else bool(set(hop.modalities) & set(trace.context_modalities)))

    # strict channel analysis: D can reach the hop only through upstream U
    channels = []
    for d in trace.agents:
        if d.raw_inputs is None or not d.handoff_from:
            continue
        if available[d.agent_id]:
            continue                                  # D sees the raw evidence itself
        ups = [u for u in d.handoff_from if trace.agent(u) is not None]
        for u in ups:
            if surfaced[u]:
                channels.append({"from": u, "to": d.agent_id, "outcome": "carried" if surfaced[d.agent_id] else "dropped"})
        if surfaced[d.agent_id] and not any(surfaced[u] for u in ups):
            channels.append({"from": None, "to": d.agent_id, "outcome": "unsupported_surface"})

    if in_context is False:
        status = "NO_EVIDENCE"
    elif not any(surfaced.values()):
        status = "NOT_SURFACED"
    elif not by_stage.get(last, False):
        status = "LOST_IN_HANDOFF"
    else:
        status = "REACHED"
    first = next((s for s in stages if by_stage[s]), None)
    return {
        "hop_index": hop.index, "items": texts, "modalities": hop.modalities, "in_context": in_context,
        "status": status, "first_stage_surfaced": first, "surfaced_by_stage": by_stage,
        "surfaced_by_agent": surfaced, "match_kind": match_kind, "raw_available": available,
        "final_text_mentions": mentions_all(texts, trace.final_text), "channels": channels,
    }


_STOP = {"the", "and", "with", "from", "that", "this", "film", "show", "song", "team", "club", "year"}


def committed(agent_answer: str, gold: GoldQuestion) -> bool:
    """Did the agent COMMIT to the gold answer in its answer field (not merely mention it)?"""
    if is_abstaining_text(agent_answer) or not mentions_all(gold.answers, agent_answer):
        return False
    n = n_tokens(agent_answer)
    if any(is_weak_gold(g) for g in gold.answers) and n > 8:
        return False          # '2008' inside a long sentence is not a commitment
    if gold.top_type == "Compare" and gold.answer_is_hop_item:
        longest = max(n_tokens(g) for g in gold.answers)
        if n > 2 * longest + 3:
            return False      # a sentence identifying the compared item is hop extraction, not the comparison
    return True


def near_miss(agent_answer: str, gold: GoldQuestion, question: str) -> bool:
    """Short, non-abstaining answer sharing a distinctive token with the gold but not matching it
    (e.g. 'Jim Caviezel' vs 'James Patrick Caviezel'). Listed for alias review; never changes a label."""
    if is_abstaining_text(agent_answer) or n_tokens(agent_answer) > 8 or mentions_all(gold.answers, agent_answer):
        return False
    a_toks = set(normalize(agent_answer).split())
    q_toks = set(normalize(question).split())
    for g in gold.answers:
        distinctive = {t for t in normalize(g).split() if len(t) >= 4 and t not in _STOP and t not in q_toks}
        if distinctive & a_toks:
            return True
    return False


def score_trace(trace: Trace, gold: GoldQuestion) -> dict:
    preds = split_prediction(trace.final_answer, len(gold.answers))
    em, f1 = mmqa_em_f1(preds, gold.answers)
    final_abstain = is_abstaining_text(trace.final_answer)
    answer_hit = {a.agent_id: committed(a.answer, gold) for a in trace.agents}
    weak = any(is_weak_gold(g) for g in gold.answers)
    compare_hop = gold.top_type == "Compare" and gold.answer_is_hop_item
    mention_ok = not (gold.answer_in_question or gold.uninformative_answer or weak or compare_hop)
    raw_mention = {a.agent_id: mentions_all(gold.answers, a.text) for a in trace.agents}
    text_hit = {aid: mention_ok and v for aid, v in raw_mention.items()}
    final_contains = (not final_abstain) and mentions_all(gold.answers, trace.final_answer)

    hops = [score_hop(h, trace) for h in gold.latent_hops]
    hop_statuses = [h["status"] for h in hops]
    stages = sorted({a.stage for a in trace.agents})

    if em == 1.0:
        label = "CORRECT"
    elif final_contains:
        label = "SURFACE_FORM"
    elif any(answer_hit.values()):
        label = "AGGREGATION_LOSS"
    elif any(text_hit.values()):
        label = "COMMITMENT_FAILURE"
    elif hops:
        worst = min(hop_statuses, key=HOP_STATUS_ORDER.index)
        label = {"NO_EVIDENCE": "NO_EVIDENCE", "NOT_SURFACED": "EXTRACTION_FAILURE",
                 "LOST_IN_HANDOFF": "HANDOFF_LOSS", "REACHED": "COMPOSITION_FAILURE"}[worst]
    elif gold.compositional:
        label = "COMPOSITION_FAILURE"             # every hop was given in the question
    elif trace.context_modalities is not None and gold.answer_modalities and \
            not set(gold.answer_modalities) & set(trace.context_modalities):
        label = "NO_EVIDENCE"
    else:
        label = "EXTRACTION_FAILURE"

    return {
        "pipeline": trace.pipeline, "qid": trace.qid, "question": gold.question, "type": gold.qtype,
        "top_type": gold.top_type, "compositional": gold.compositional, "gold": gold.answers,
        "prediction": trace.final_answer, "prediction_items": preds, "em": em, "f1": f1,
        "final_abstain": final_abstain, "label": label,
        "n_hops": len(gold.hops), "n_latent_hops": len(hops), "given_hops": len(gold.hops) - len(hops),
        "hop_statuses": hop_statuses, "hops": hops,
        "agent_answer_hit": answer_hit, "agent_text_mentions_gold": text_hit,
        "gold_in_question": gold.answer_in_question, "uninformative_gold": gold.uninformative_answer,
        "weak_gold": weak, "compare_answer_is_hop": compare_hop,
        "unscored_gold_mentions": [aid for aid, v in raw_mention.items() if v and not mention_ok],
        "near_miss_agents": [a.agent_id for a in trace.agents if near_miss(a.answer, gold, gold.question)],
        "near_miss_answers": sorted({a.answer for a in trace.agents if near_miss(a.answer, gold, gold.question)}),
        "correct_without_bridge": em == 1.0 and any(s in ("NOT_SURFACED", "NO_EVIDENCE") for s in hop_statuses),
        "context_modalities": trace.context_modalities, "stages": stages,
        "synthesis_stages": sorted({a.stage for a in trace.agents if not a.decision}),
        "agents": [a.agent_id for a in trace.agents],
    }


# ----------------------------------------------------------------------------- aggregation
def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def summarize(rows: List[dict]) -> dict:
    n = len(rows)
    labels = Counter(r["label"] for r in rows)
    hops = [h for r in rows for h in r["hops"]]
    in_ctx = [h for h in hops if h["in_context"] is not False]
    stages = sorted({s for r in rows for s in r["stages"]})
    stage_recall = {f"S{s}": _rate(sum(h["surfaced_by_stage"].get(s, False) for h in in_ctx), len(in_ctx)) for s in stages}
    stage_recall["final_text"] = _rate(sum(h["final_text_mentions"] for h in in_ctx), len(in_ctx))
    first_surfaced = [h for h in in_ctx if h["first_stage_surfaced"] is not None]
    synth = sorted({s for r in rows for s in r["synthesis_stages"]})
    early = [h for h in in_ctx if h["surfaced_by_stage"].get(synth[0], False)] if synth else []
    lost = [h for h in early if not h["surfaced_by_stage"].get(synth[-1], False)]

    agent_av, agent_hit = Counter(), Counter()
    for h in in_ctx:
        for aid, av in h["raw_available"].items():
            if av:
                agent_av[aid] += 1
                agent_hit[aid] += h["surfaced_by_agent"][aid]
    chan = defaultdict(Counter)
    for h in hops:
        for c in h["channels"]:
            chan[f"{c['from']}->{c['to']}"][c["outcome"]] += 1
    strict_carried = sum(c["carried"] for c in chan.values())
    strict_dropped = sum(c["dropped"] for c in chan.values())

    wrong = [r for r in rows if r["label"] != "CORRECT"]
    return {
        "n_questions": n,
        "EM": _rate(sum(r["em"] for r in rows), n),
        "mean_F1": round(sum(r["f1"] for r in rows) / n, 4) if n else None,
        "final_abstain_rate": _rate(sum(r["final_abstain"] for r in rows), n),
        "labels": {k: labels.get(k, 0) for k in LABEL_ORDER},
        "label_share_of_errors": {k: _rate(labels.get(k, 0), len(wrong)) for k in LABEL_ORDER if k != "CORRECT"},
        "questions_with_latent_hops": sum(1 for r in rows if r["n_latent_hops"]),
        "latent_hops": len(hops), "latent_hops_in_context": len(in_ctx),
        "hop_status": dict(Counter(h["status"] for h in hops)),
        "bridge_recall_by_stage": stage_recall,
        "bridge_recall_any_agent": _rate(len(first_surfaced), len(in_ctx)),
        "handoff_loss_stage_level": {"surfaced_at_first_stage": len(early), "lost_by_last_stage": len(lost),
                                     "rate": _rate(len(lost), len(early))},
        "handoff_loss_strict_channel": {"carried": strict_carried, "dropped": strict_dropped,
                                        "rate": _rate(strict_dropped, strict_carried + strict_dropped),
                                        "unsupported_surfacing": sum(c["unsupported_surface"] for c in chan.values()),
                                        "by_channel": {k: dict(v) for k, v in sorted(chan.items())}},
        "agent_recall_when_raw_available": {a: {"available": agent_av[a], "surfaced": agent_hit[a],
                                                "recall": _rate(agent_hit[a], agent_av[a])} for a in sorted(agent_av)},
        "correct_without_bridge": sum(r["correct_without_bridge"] for r in rows),
        "aggregation_loss_qids": [r["qid"] for r in rows if r["label"] == "AGGREGATION_LOSS"],
        "unscored_gold_mentions": sum(1 for r in rows if r["unscored_gold_mentions"] and r["label"] != "CORRECT"),
        "near_misses_for_alias_review": [{"qid": r["qid"], "gold": r["gold"], "answers": r["near_miss_answers"]}
                                         for r in rows if r["near_miss_answers"] and r["label"] != "CORRECT"],
    }
