"""Readers that turn pipeline outputs into one common trace format.

A Trace is a list of AgentOutputs (the intermediate agents) plus the final answer. Each agent records:

* stage            1 = modality experts, 2 = cross-modal / anchor agents, 3 = verifier etc.
* answer           the field the agent committed to as its answer
* text             everything the agent wrote (answer + insight/reason/raw output)
* raw_inputs       modalities whose RAW evidence this agent saw (None = unknown for this pipeline)
* handoff_from     upstream agents whose output this agent received

`raw_inputs` and `handoff_from` encode the pipeline's information-flow graph. In MAMMQA, the
cross-modal agent anchored on modality X never sees X's raw evidence, only X's Stage-I insight,
so a hop that lives in X can reach it only through that hand-off. That is what makes strict
hand-off loss measurable.

Add a reader here (and register it in READERS) to audit a new pipeline.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

MODALITIES = ("text", "table", "image")
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


@dataclass
class AgentOutput:
    agent_id: str
    stage: int
    anchor: Optional[str]
    answer: str
    text: str
    raw_inputs: Optional[List[str]]
    handoff_from: List[str] = field(default_factory=list)
    decision: bool = False     # verifier/judge-style agent: treated as part of the decision, not synthesis


@dataclass
class Trace:
    pipeline: str
    qid: str
    question: str
    agents: List[AgentOutput]
    final_answer: str
    final_text: str
    context_modalities: Optional[List[str]]
    meta: dict = field(default_factory=dict)

    def agent(self, agent_id: str) -> Optional[AgentOutput]:
        return next((a for a in self.agents if a.agent_id == agent_id), None)

    def to_dict(self) -> dict:
        return asdict(self)


def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return " ".join(_s(v) for v in x)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _last_answer_tag(raw: str) -> str:
    """The agent's committed answer = its last non-empty <answer> tag. v0.2: an output with NO tag commits
    to nothing (''); v0.1 used the whole output, which turned reasoning mentions into 'commitments'
    (24% of agent outputs in the 300-question run had no tag)."""
    tags = [t.strip() for t in _ANSWER_TAG.findall(raw or "") if t.strip()]
    return tags[-1] if tags else ""


# ----------------------------------------------------------------------------- CV-MAMMQA pilot
def read_cv_pilot(shared_path: Path, final_path: Path, pipeline: str) -> List[Trace]:
    """Kaggle CV-MAMMQA provided-context pilot. `predictions_shared.jsonl` holds the shared Stage I/II
    outputs; `final_path` is predictions_mammqa.jsonl (baseline aggregator) or predictions_cv_mammqa.jsonl.
    Context = gold supporting docs only; images were shown with their Wikipedia titles as text."""
    finals = {r["qid"]: r for r in read_jsonl(final_path)}
    traces = []
    for r in read_jsonl(shared_path):
        s1, s2 = r.get("stage1", {}), r.get("stage2", {})
        ctx = sorted(m for m, v in s1.items() if v.get("raw_output") != "NOT_RUN")
        agents = []
        for m, v in s1.items():
            if v.get("raw_output") == "NOT_RUN":
                continue
            agents.append(AgentOutput(f"S1.{m}", 1, m, _s(v.get("answer")),
                                      "\n".join([_s(v.get("answer")), _s(v.get("insight")), _s(v.get("raw_output"))]),
                                      raw_inputs=[m]))
        for m, v in s2.items():
            agents.append(AgentOutput(f"S2.{m}", 2, m, _s(v.get("answer")),
                                      "\n".join([_s(v.get("answer")), _s(v.get("reason")),
                                                 _s(v.get("evidence_reference")), _s(v.get("raw_output"))]),
                                      raw_inputs=[c for c in ctx if c != m],
                                      handoff_from=[f"S1.{m}"] if f"S1.{m}" in {a.agent_id for a in agents} else []))
        fin = finals.get(r["qid"], {})
        traces.append(Trace(pipeline, r["qid"], r["question"], agents, _s(fin.get("prediction")),
                            _s(fin.get("raw_output")), ctx,
                            meta={"source": str(final_path.name), "fingerprint": r.get("fingerprint", "")[:16],
                                  "context": "gold supporting docs; image titles shown as text"}))
    return traces


# ----------------------------------------------------------------------------- official-prompt MAMMQA
def read_mammqa_official(path: Path, pipeline: str) -> List[Trace]:
    """mammqa_full_outputs/mammqa_predictions.jsonl: released MAMMQA prompts (<insights>/<answer> tags).
    Information flow follows agents.py: X_cross gets X's Stage-I insight plus the raw other modalities."""
    traces = []
    for r in read_jsonl(path):
        s1 = r.get("stage1", {})
        ctx = sorted(m for m, v in s1.items() if (v or {}).get("status") == "available")
        agents = []
        for m in MODALITIES:
            v = s1.get(m) or {}
            if v.get("status") != "available":
                continue
            out = _s(v.get("output"))
            agents.append(AgentOutput(f"S1.{m}", 1, m, _last_answer_tag(out), out, raw_inputs=[m]))
        ids = {a.agent_id for a in agents}
        for m in MODALITIES:
            out = _s((r.get("stage2") or {}).get(m))
            if not out:
                continue
            agents.append(AgentOutput(f"S2.{m}", 2, m, _last_answer_tag(out), out,
                                      raw_inputs=[c for c in ctx if c != m],
                                      handoff_from=[f"S1.{m}"] if f"S1.{m}" in ids else []))
        traces.append(Trace(pipeline, r["id"], r["question"], agents, _s(r.get("prediction")),
                            _s(r.get("final_raw")), ctx,
                            meta={"source": path.name, "file_gold": r.get("gold_answer")}))
    return traces


# ----------------------------------------------------------------------------- HM-RAG oracle arm
def read_hm_oracle(path: Path, pipeline: str) -> List[Trace]:
    """HM-RAG + MAMMQA smoke run, MAMMQA_ORACLE arm (gold docs given as retrieval). The cross-modal
    agents' exact raw inputs are not recorded in the trace, so raw_inputs=None: strict hand-off loss
    is not computed for this pipeline, only the stage-level version."""
    traces = []
    for r in read_jsonl(path):
        s1, cm = r.get("stage1") or {}, r.get("cross_modal") or {}
        ctx = sorted(m for m in s1 if m in MODALITIES)
        agents = []
        for m, v in s1.items():
            v = v or {}
            agents.append(AgentOutput(f"S1.{m}", 1, m, _s(v.get("candidate_answer")),
                                      "\n".join([_s(v.get("candidate_answer")), _s(v.get("evidence_summary")), _s(v.get("_raw"))]),
                                      raw_inputs=[m]))
        s1_ids = [a.agent_id for a in agents]
        for m, v in cm.items():
            v = v or {}
            agents.append(AgentOutput(f"S2.{m}", 2, m, _s(v.get("candidate_answer")),
                                      "\n".join([_s(v.get("candidate_answer")), _s(v.get("evidence_summary")), _s(v.get("_raw"))]),
                                      raw_inputs=None, handoff_from=list(s1_ids)))
        ver = r.get("verification") or {}
        if ver:
            agents.append(AgentOutput("S3.verifier", 3, None, _s(ver.get("best_answer")), _s(ver.get("_raw")),
                                      raw_inputs=None, handoff_from=[a.agent_id for a in agents if a.stage == 2],
                                      decision=True))
        fd = r.get("final_decision") or {}
        traces.append(Trace(pipeline, r["qid"], r["question"], agents, _s(r.get("final_answer")),
                            _s(fd.get("_raw")) or _s(r.get("final_answer")), ctx,
                            meta={"source": path.name, "retrieval_rounds": r.get("retrieval_rounds")}))
    return traces


READERS: Dict[str, Callable[..., List[Trace]]] = {
    "cv_pilot": read_cv_pilot,
    "mammqa_official": read_mammqa_official,
    "hm_oracle": read_hm_oracle,
}
