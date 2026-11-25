"""Tests for the hop audit.  Run from the hop_audit folder:  python -m unittest discover -s tests -v

Unit tests use synthetic MMQA rows and traces, so they need no data. The integration tests run only
if the project's MMQA dev file and the Kaggle CV pilot outputs are present.
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from hop_audit import matching                                     # noqa: E402
from hop_audit.gold import build_gold, load_gold                   # noqa: E402
from hop_audit.matching import mention_level, mmqa_em_f1, split_prediction  # noqa: E402
from hop_audit.scoring import score_trace, summarize               # noqa: E402
from hop_audit.traces import AgentOutput, Trace, read_cv_pilot     # noqa: E402


def ia(answer, modality, doc="d1"):
    inst = [{"doc_id": doc, "doc_part": "image"}] if modality == "image" else []
    txt = [{"doc_id": doc, "part": "text", "start_byte": 0, "text": answer}] if modality == "text" else []
    return {"answer": answer, "type": "string", "modality": modality, "text_instances": txt,
            "table_indices": [[0, 0]] if modality == "table" else [], "image_instances": inst}


def row(qid, question, qtype, answers, hops, support=("text", "table")):
    return {"qid": qid, "question": question,
            "answers": [{"answer": a, "type": "string", "modality": "table"} for a in answers],
            "metadata": {"type": qtype, "intermediate_answers": hops, "pseudo_language_question": ""},
            "supporting_context": [{"doc_id": f"s{i}", "doc_part": m} for i, m in enumerate(support)]}


# A Compose(TextQ,TableQ) question: bridge 'The Bad News Bears' lives in the table, answer '1976' in text.
COMPOSE = build_gold(row("q1", "When was the movie that had Ben Piazza as Bob Whitewood released?",
                         "Compose(TextQ,TableQ)", [1976.0], [[ia("The Bad News Bears", "table")]]))


def mammqa_trace(s1_text, s1_table, s2_text, s2_table, final, s1_text_ans="INSUFFICIENT",
                 s1_table_ans="INSUFFICIENT", s2_text_ans="INSUFFICIENT", s2_table_ans="INSUFFICIENT"):
    ctx = ["table", "text"]
    agents = [
        AgentOutput("S1.text", 1, "text", s1_text_ans, s1_text, ["text"]),
        AgentOutput("S1.table", 1, "table", s1_table_ans, s1_table, ["table"]),
        AgentOutput("S2.text", 2, "text", s2_text_ans, s2_text, ["table"], ["S1.text"]),
        AgentOutput("S2.table", 2, "table", s2_table_ans, s2_table, ["text"], ["S1.table"]),
    ]
    return Trace("t", "q1", COMPOSE.question, agents, final, final, ctx)


class TestMatching(unittest.TestCase):
    def test_exact_and_whole_token(self):
        self.assertEqual(mention_level("Mask", 'He starred in "Mask" (1985).'), "exact")
        self.assertIsNone(mention_level("Mask", "a masked man"))

    def test_disambiguator_and_accents(self):
        self.assertEqual(mention_level("Mask (1985 film)", "the film Mask"), "exact")
        self.assertEqual(mention_level("Dženis Beganović", "Dzenis Beganovic played"), "exact")

    def test_numbers(self):
        self.assertEqual(mention_level(1976.0, "released in 1976"), "exact")
        self.assertEqual(mention_level("eight", "there are 8 seats"), "exact")

    def test_fuzzy_long_names_only(self):
        self.assertEqual(mention_level("Tell Me That You Love Me, Junie Moon", "Tell Me You Love Me, Junie Moon poster"), "fuzzy")
        self.assertIsNone(mention_level("Gus", "Gust of wind"))

    def test_alias(self):
        matching.ALIASES.clear()
        self.assertIsNone(mention_level("James Patrick Caviezel", "Jim Caviezel played Reese"))
        matching.ALIASES[matching.normalize("James Patrick Caviezel")] = ["Jim Caviezel"]
        self.assertEqual(mention_level("James Patrick Caviezel", "Jim Caviezel played Reese"), "exact")
        matching.ALIASES.clear()

    def test_official_em(self):
        self.assertEqual(mmqa_em_f1(["The Bad News Bears"], ["Bad News Bears"])[0], 1.0)
        self.assertEqual(mmqa_em_f1(["1976"], [1976.0])[0], 1.0)
        self.assertEqual(mmqa_em_f1(["Mil Mi-8", "Mil Mi-24"], ["Mil Mi-24", "Mil Mi-8"])[0], 1.0)
        em, f1 = mmqa_em_f1(["Mil Mi-8"], ["Mil Mi-24", "Mil Mi-8"])
        self.assertEqual(em, 0.0); self.assertGreater(f1, 0.0)

    def test_split_prediction(self):
        self.assertEqual(split_prediction("Washington, D.C.", 1), ["Washington, D.C"])
        self.assertEqual(split_prediction("A, B and C", 3), ["A", "B", "C"])

    def test_abstaining_sentences(self):
        self.assertTrue(matching.is_abstaining_text("There is insufficient information about other horses."))
        self.assertTrue(matching.is_abstaining_text("FINAL: INSUFFICIENT"))
        self.assertFalse(matching.is_abstaining_text("Victor Espinoza rode American Pharoah."))

    def test_weak_golds(self):
        for g in ["1976", "yes", "8", "horses"]:
            self.assertTrue(matching.is_weak_gold(g), g)
        for g in ["Essendon", "James Taylor"]:
            self.assertFalse(matching.is_weak_gold(g), g)


class TestGold(unittest.TestCase):
    def test_given_vs_latent(self):
        g = build_gold(row("q2", 'Which came first: "Nightwing" or the movie with half a face on the poster?',
                           "Compare(Compose(TableQ,ImageQ),TableQ)", ["Junie Moon"],
                           [[ia("Junie Moon", "image")], [ia("Nightwing", "table")]]))
        self.assertEqual([h.latent for h in g.hops], [True, False])
        self.assertTrue(g.answer_is_hop_item)

    def test_intersect_requires_overlapping_item(self):
        g = build_gold(row("q3", "Which team ...?", "Intersect(TableQ,TextQ)", ["Essendon"],
                           [[ia("Essendon Football Club", "text"), ia("Brisbane Lions", "text")], [ia("Essendon", "table")]]))
        self.assertEqual(g.hops[0].required_texts, ["Essendon Football Club"])

    def test_compose(self):
        self.assertTrue(COMPOSE.compositional)
        self.assertEqual(COMPOSE.answers, ["1976"])
        self.assertEqual(COMPOSE.latent_hops[0].modalities, ["table"])


class TestLabels(unittest.TestCase):
    def test_correct(self):
        t = mammqa_trace("", "Bad News Bears row", "The Bad News Bears, 1976", "", "1976", s2_text_ans="1976")
        self.assertEqual(score_trace(t, COMPOSE)["label"], "CORRECT")

    def test_aggregation_loss(self):
        t = mammqa_trace("", "The Bad News Bears", "The Bad News Bears came out in 1976", "", "INSUFFICIENT", s2_text_ans="1976")
        self.assertEqual(score_trace(t, COMPOSE)["label"], "AGGREGATION_LOSS")

    def test_extraction_failure(self):
        t = mammqa_trace("nothing", "no relevant row", "nothing", "nothing", "INSUFFICIENT")
        r = score_trace(t, COMPOSE)
        self.assertEqual(r["label"], "EXTRACTION_FAILURE")
        self.assertEqual(r["hop_statuses"], ["NOT_SURFACED"])

    def test_handoff_loss_and_strict_channel(self):
        t = mammqa_trace("nothing", "Piazza played Bob Whitewood in The Bad News Bears", "nothing", "nothing", "INSUFFICIENT")
        r = score_trace(t, COMPOSE)
        self.assertEqual(r["label"], "HANDOFF_LOSS")
        self.assertEqual(r["hops"][0]["channels"], [{"from": "S1.table", "to": "S2.table", "outcome": "dropped"}])

    def test_composition_failure(self):
        t = mammqa_trace("", "The Bad News Bears", "", "The Bad News Bears, but no date", "1980")
        r = score_trace(t, COMPOSE)
        self.assertEqual(r["label"], "COMPOSITION_FAILURE")
        self.assertEqual(r["hops"][0]["channels"][0]["outcome"], "carried")

    def test_weak_gold_mention_is_not_commitment(self):
        # '1976' appears inside a long abstaining sentence: not a commitment, so the hop logic decides
        t = mammqa_trace("", "The Bad News Bears", "", "The Bad News Bears (1976) is listed, but I cannot determine the answer.",
                         "INSUFFICIENT")
        self.assertEqual(score_trace(t, COMPOSE)["label"], "COMPOSITION_FAILURE")

    def test_commitment_failure_strong_gold(self):
        g = build_gold(row("q4", "Who sang backup on that song?", "Intersect(ImageListQ,TextQ)", ["James Taylor"],
                           [[ia("Joni Mitchell", "text"), ia("James Taylor", "text")]], support=("text",)))
        agents = [AgentOutput("S1.text", 1, "text", "INSUFFICIENT", "It only names Joni Mitchell and James Taylor.", ["text"]),
                  AgentOutput("S2.text", 2, "text", "INSUFFICIENT", "Unclear.", [], ["S1.text"])]
        r = score_trace(Trace("t", "q4", g.question, agents, "INSUFFICIENT", "", ["text"]), g)
        self.assertEqual(r["label"], "COMMITMENT_FAILURE")

    def test_surface_form(self):
        t = mammqa_trace("", "", "", "", "It was released in 1976")
        self.assertEqual(score_trace(t, COMPOSE)["label"], "SURFACE_FORM")

    def test_no_evidence(self):
        g = build_gold(row("q5", "What colour is the logo?", "Compose(ImageQ,TableQ)", ["blue"],
                           [[ia("Calgary Boomers", "image")]], support=("table", "image")))
        agents = [AgentOutput("S1.table", 1, "table", "INSUFFICIENT", "nothing", ["table"])]
        r = score_trace(Trace("t", "q5", g.question, agents, "INSUFFICIENT", "", ["table"]), g)
        self.assertEqual(r["label"], "NO_EVIDENCE")

    def test_decision_agent_does_not_count_as_reach(self):
        agents = [AgentOutput("S1.table", 1, "table", "", "The Bad News Bears", ["table"]),
                  AgentOutput("S2.text", 2, "text", "", "nothing", None, ["S1.table"]),
                  AgentOutput("S3.verifier", 3, None, "", "The Bad News Bears", None, ["S2.text"], decision=True)]
        r = score_trace(Trace("t", "q1", COMPOSE.question, agents, "ABSTAIN", "", ["table", "text"]), COMPOSE)
        self.assertEqual(r["hop_statuses"], ["LOST_IN_HANDOFF"])

    def test_summary_runs(self):
        rows = [score_trace(mammqa_trace("", "The Bad News Bears", "", "", "INSUFFICIENT"), COMPOSE)]
        s = summarize(rows)
        self.assertEqual(s["handoff_loss_strict_channel"]["dropped"], 1)
        self.assertEqual(s["bridge_recall_by_stage"]["S1"], 1.0)


ROOT = HERE.parent
DEV = ROOT / "MAMMQA-main" / "Codes" / "multimodalqa-dataset" / "MMQA_dev.jsonl.gz"
CV = ROOT / "cv_mammqa_16" / "predictions"


@unittest.skipUnless(DEV.exists() and (CV / "predictions_shared.jsonl").exists(), "project data not present")
class TestCVPilotIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        matching.load_aliases(HERE / "aliases.json")
        cls.traces = read_cv_pilot(CV / "predictions_shared.jsonl", CV / "predictions_mammqa.jsonl", pipeline="b")
        cls.gold = load_gold(DEV, {t.qid for t in cls.traces})
        cls.rows = {t.qid[:8]: score_trace(t, cls.gold[t.qid]) for t in cls.traces}

    def test_em_matches_kaggle_summary(self):
        self.assertEqual(sum(r["em"] for r in self.rows.values()), 3)   # metrics/summary.csv: EM 0.1875

    def test_known_cases(self):
        self.assertEqual(self.rows["64dabbb9"]["label"], "COMMITMENT_FAILURE")  # 'James Taylor' named, INSUFFICIENT
        self.assertEqual(self.rows["ad36bba9"]["label"], "AGGREGATION_LOSS")    # Stage I text: 'Jim Caviezel'
        self.assertEqual(self.rows["782b7cd2"]["label"], "AGGREGATION_LOSS")    # full cast in Stage I, final one name
        self.assertEqual(self.rows["cdc57d98"]["hop_statuses"], ["REACHED"])    # Dances with Wolves found

    @classmethod
    def tearDownClass(cls):
        matching.ALIASES.clear()


if __name__ == "__main__":
    unittest.main()
