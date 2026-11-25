import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for host integration")
class ActivationRuntimeTests(unittest.TestCase):
    def test_temperature_changes_decision_and_committed_prediction_is_exact(self):
        import torch
        from torch.utils.data import TensorDataset

        from src.eval_activation import collect, gains, probe_runtime_summary
        from src.run_har import OPTIONAL, PRIMARY

        def embedding(value):
            return lambda signal: torch.full((len(signal), 128), value)

        def fuse(slots, mask):
            optional = slots[:, 1, 0] * mask[:, 1]
            return torch.stack((optional, torch.zeros_like(optional)), dim=1)

        host = SimpleNamespace(encoders={PRIMARY: embedding(0.0), OPTIONAL: embedding(-4.0)}, fuse=fuse)
        probes = {OPTIONAL: embedding(2.0)}
        dataset = TensorDataset(torch.zeros((1, 2, 3, 128)), torch.tensor([1]))
        windows = collect(host, probes, dataset)
        expected = [(1.0, 1, 1.0), (2.0, 0, 0.0)]
        for temperature, activations, accuracy in expected:
            with self.subTest(temperature=temperature):
                scores = gains(windows, windows.probe, temperature)
                result = probe_runtime_summary(host, probes, dataset, windows, scores, 0.3, temperature)
                self.assertEqual(result["activations"], activations)
                self.assertEqual(result["accuracy"], accuracy)
                self.assertEqual(result["controller_verified_windows"], 1)

    def test_six_axis_acceleration_keeps_three_axis_gyroscope(self):
        import torch

        from src.model import Host, Probe, WindowInference
        from src.run_har import OPTIONAL, PRIMARY

        torch.set_num_threads(1)
        host = Host(acceleration_channels=6).eval()
        probe = Probe(host.encoders[OPTIONAL]).eval()
        signal = torch.ones(2, 6, 128)
        engine = WindowInference(host, {OPTIONAL: probe}, signal)
        self.assertEqual(engine.signal(PRIMARY).shape, (1, 6, 128))
        self.assertEqual(engine.signal(OPTIONAL).shape, (1, 3, 128))
        with torch.inference_mode():
            full = {name: engine.encode(name) for name in (PRIMARY, OPTIONAL)}
            self.assertEqual(engine.logits(full), host(signal.unsqueeze(0))[0].tolist())
            self.assertEqual(engine.probe(OPTIONAL).shape, (1, 128))

    def test_benefit_budget_uses_signed_threshold_and_preserves_ties(self):
        import torch

        from src.try_benefit import budget_threshold

        scores = torch.tensor([-0.5, -0.2, -0.2, 0.1], dtype=torch.float64)
        self.assertEqual(int((scores > budget_threshold(scores, 2)).sum()), 1)
        self.assertEqual(int((scores > budget_threshold(scores, 4)).sum()), 4)

    def test_benefit_policy_runs_optional_encoder_only_when_selected(self):
        import torch

        from src.run_har import OPTIONAL, PRIMARY
        from src.try_benefit import benefit_policy

        calls = []

        def encode(modality):
            calls.append(modality)
            return modality

        engine = SimpleNamespace(encode=encode,
                                 logits=lambda active: [0.0, 1.0] if OPTIONAL in active else [1.0, 0.0])
        selector = SimpleNamespace(utility=lambda inputs: torch.tensor([0.2]))
        self.assertEqual(benefit_policy(engine, selector, 0.3, use_probe=False), (0, 1, 0))
        self.assertEqual(calls, [PRIMARY])
        self.assertEqual(benefit_policy(engine, selector, 0.1, use_probe=False), (1, 2, 0))
        self.assertEqual(calls, [PRIMARY, PRIMARY, OPTIONAL])
