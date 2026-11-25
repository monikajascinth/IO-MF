import math
import unittest

from src.controllers.iomf import Timing, distribution, entropy, infer


class Engine:
    def __init__(self, probes):
        self.probes = probes
        self.calls = []

    def encode(self, modality):
        self.calls.append(("encode", modality))
        return 0 if modality == "p" else -8

    def probe(self, modality):
        self.calls.append(("probe", modality))
        return self.probes[modality]

    def logits(self, embeddings):
        return sum(embeddings.values()), 0


class ControllerTests(unittest.TestCase):
    def test_entropy_extremes(self):
        self.assertAlmostEqual(entropy(distribution([3, 3])), math.log(2))
        self.assertEqual(entropy(distribution([1000, -1000])), 0)

    def test_zero_utility_and_empty_optional(self):
        engine = Engine({"a": 0})
        self.assertEqual(infer(engine, ["p"], ["a"], {"a": 0}, 0).active, ("p",))
        self.assertEqual(infer(engine, ["p"], [], {}, 0).reason, "exhausted")
        self.assertNotIn(("encode", "a"), engine.calls)

    def test_utility_ranking_and_exact_prediction(self):
        engine = Engine({"a": 8, "b": 4})
        result = infer(engine, ["p"], ["a", "b"], {"a": 0.5, "b": 0}, 1)
        self.assertEqual(result.active, ("p", "b"))
        self.assertEqual(result.prediction, 1)
        self.assertEqual(engine.calls.count(("encode", "p")), 1)
        self.assertEqual(engine.calls.count(("encode", "b")), 1)

    def test_fixed_ties(self):
        engine = Engine({"a": 8, "b": 8})
        self.assertEqual(infer(engine, ["p"], ["b", "a"], {"a": 0, "b": 0}, 0).active,
                         ("p", "b"))

    def test_reservation_skips_one_candidate_and_scans_next(self):
        engine = Engine({"a": 4, "b": 20, "c": 8})
        timing = Timing(2, {"a": 1, "b": 4, "c": 1}, {"a": 5, "b": 1, "c": 4})
        result = infer(engine, ["p"], ["a", "b", "c"], dict.fromkeys("abc", 0), 0,
                       deadline=10, timing=timing)
        self.assertEqual(result.active, ("p", "c"))
        self.assertEqual(result.residual, 2)
        self.assertEqual([(e.action, e.modality) for e in result.trace],
                         [("primary", "p"), ("probe", "a"), ("skip", "b"),
                          ("probe", "c"), ("commit", "c"), ("skip", "b")])

    def test_infeasible_and_exact_budget(self):
        engine = Engine({"a": 8})
        timing = Timing(2, {"a": 1}, {"a": 2})
        args = (engine, ["p"], ["a"], {"a": 0}, 0)
        self.assertIsNone(infer(*args, deadline=1, timing=timing).prediction)
        self.assertEqual(engine.calls, [])
        self.assertEqual(infer(*args, deadline=2, timing=timing).active, ("p",))
        result = infer(*args, deadline=5, timing=timing)
        self.assertEqual(result.active, ("p", "a"))
        self.assertEqual(result.residual, 0)

    def test_no_positive_leader_debits_probe(self):
        engine = Engine({"a": 0})
        result = infer(engine, ["p"], ["a"], {"a": 0}, 0, deadline=5,
                       timing=Timing(2, {"a": 1}, {"a": 2}))
        self.assertEqual(result.active, ("p",))
        self.assertEqual(result.residual, 2)

    def test_second_round_can_admit_previously_skipped_candidate(self):
        engine = Engine({"a": 4, "b": -8, "c": 8})
        timing = Timing(2, {"a": 1, "b": 4, "c": 1}, {"a": 5, "b": 1, "c": 1})
        result = infer(engine, ["p"], ["a", "b", "c"], dict.fromkeys("abc", 0), 0,
                       deadline=10, timing=timing)
        self.assertEqual(result.active, ("p", "c", "b"))
        self.assertEqual(result.residual, 0)
        self.assertEqual([e.action for e in result.trace],
                         ["primary", "probe", "skip", "probe", "commit", "probe", "commit"])
