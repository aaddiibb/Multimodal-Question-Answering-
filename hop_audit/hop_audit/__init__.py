"""Hop-level audit for multi-agent multimodal QA on MultiModalQA.

Scores where gold intermediate answers (bridge entities) appear, survive or disappear across
the agents of a pipeline, and labels every wrong answer with the earliest point of failure.
"""
from .gold import GoldQuestion, Hop, HopItem, build_gold, load_gold
from .matching import mention_level, mmqa_em_f1, normalize
from .scoring import LABEL_ORDER, score_trace, summarize
from .traces import READERS, AgentOutput, Trace

__version__ = "0.2.0"   # 0.2: en-dash normalisation; untagged agent outputs commit to nothing
