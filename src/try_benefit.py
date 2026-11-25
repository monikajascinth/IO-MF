"""Try a learned gyroscope-benefit policy alongside manuscript IO-MF."""

import argparse
import json
from bisect import bisect_right
from math import inf, nextafter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.compare_probes import ARTICLE_THRESHOLD, threshold_for_count
from src.eval_activation import collect, decision_summary, fit_temperature, gains
from src.eval_har import controller_policy, exact_policy, load_run, profile, summarize
from src.run_har import OPTIONAL, PRIMARY, fingerprint


def features(primary, probe=None):
    sorted_logits = primary.topk(2, dim=1).values
    margin = sorted_logits[:, :1] - sorted_logits[:, 1:2]
    if probe is None:
        return torch.cat((primary, margin), dim=1)
    return torch.cat((primary, margin, probe - primary), dim=1)


class Selector(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.register_buffer("center", inputs.mean(dim=0))
        self.register_buffer("scale", inputs.std(dim=0, correction=0).clamp_min(1e-6))
        self.linear = nn.Linear(inputs.shape[1], 3, dtype=inputs.dtype)

    def forward(self, inputs):
        return self.linear((inputs - self.center) / self.scale)

    def utility(self, inputs):
        probabilities = self(inputs).softmax(dim=1)
        return probabilities[:, 2] - probabilities[:, 0]


def fit_selector(inputs, targets, seed):
    torch.manual_seed(seed)
    model = Selector(inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=0.001)
    for _ in range(100):
        optimizer.zero_grad()
        F.cross_entropy(model(inputs), targets).backward()
        optimizer.step()
    return model.eval()


def benefit_targets(windows):
    primary_correct = windows.primary.argmax(dim=1) == windows.labels
    full_correct = windows.full.argmax(dim=1) == windows.labels
    return full_correct.long() - primary_correct.long() + 1


def budget_threshold(scores, target):
    # Unlike manuscript IO-MF, a learned score can use a signed cutoff to meet a budget.
    values = sorted(scores.tolist())
    candidates = {nextafter(values[0], -inf), *values}
    return min(candidates, key=lambda threshold: (
        abs(len(values) - bisect_right(values, threshold) - target),
        len(values) - bisect_right(values, threshold) > target,
        threshold,
    ))


def summarize_policy(windows, scores, threshold, *, entropy):
    summary = decision_summary(windows, scores, threshold)
    if not entropy:
        summary["threshold_score"] = summary.pop("threshold_nats")
        summary.pop("max_activations_at_nonnegative_threshold")
    return summary


def benefit_policy(engine, selector, cutoff, use_probe):
    active = {PRIMARY: engine.encode(PRIMARY)}
    primary_logits = engine.logits(active)
    primary = torch.tensor([primary_logits], dtype=torch.float64)
    probe = None
    if use_probe:
        surrogate = engine.probe(OPTIONAL)
        probe = torch.tensor([engine.logits({**active, OPTIONAL: surrogate})], dtype=torch.float64)
    selected = selector.utility(features(primary, probe)).item() > cutoff
    if selected:
        active[OPTIONAL] = engine.encode(OPTIONAL)
    logits = engine.logits(active) if selected else primary_logits
    return max(range(len(logits)), key=logits.__getitem__), len(active), int(use_probe)


@torch.inference_mode()
def profile_policies(host, probes, test_data, selectors, use_probe, selected, thresholds, temperature, warmup):
    policies = {
        selected: lambda engine: benefit_policy(engine, selectors[selected], thresholds[selected], use_probe[selected]),
        "early_fusion": lambda engine: exact_policy(engine, include_optional=True),
    }
    labels = test_data.tensors[1].tolist()
    timing = {
        name: summarize(labels, profile(host, probes, test_data, policy, warmup))
        for name, policy in policies.items()
    }
    def calibrated(engine):
        return controller_policy(engine, thresholds["io_mf_calibrated"])

    timing["io_mf_calibrated"] = summarize(
        labels, profile(host, probes, test_data, calibrated, warmup, temperature=temperature)
    )
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--activation-rate", type=float, default=0.52)
    parser.add_argument("--profile", action="store_true", help="Measure live CPU policy execution")
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.activation_rate <= 1 or args.warmup < 0 or args.output.exists():
        parser.error("Use an activation rate in [0, 1], nonnegative warmup and a new output directory")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    host, probes, (train_data, validation_data, test_data), original = load_run(args.dataset, args.run)
    train = collect(host, probes, train_data)
    validation = collect(host, probes, validation_data)
    targets = benefit_targets(train)
    policies = {"benefit_primary": False, "benefit_probe": True}
    selectors = {
        name: fit_selector(features(train.primary, train.probe if use_probe else None),
                           targets, original["seed"])
        for name, use_probe in policies.items()
    }
    temperature = fit_temperature(validation)
    count = round(args.activation_rate * len(validation.labels))

    @torch.inference_mode()
    def scores(windows):
        return {
            "io_mf_article_threshold": gains(windows, windows.probe, 1.0),
            "io_mf_calibrated": gains(windows, windows.probe, temperature),
            "exact_entropy_calibrated": gains(windows, windows.full, temperature),
            **{name: selectors[name].utility(features(windows.primary, windows.probe if use_probe else None))
               for name, use_probe in policies.items()},
        }

    validation_scores = scores(validation)
    thresholds = {
        name: budget_threshold(values, count) if name in selectors else threshold_for_count(values.tolist(), count)
        for name, values in validation_scores.items()
    }
    thresholds["io_mf_article_threshold"] = ARTICLE_THRESHOLD
    validation_results = {
        name: summarize_policy(validation, values, thresholds[name], entropy=name not in selectors)
        for name, values in validation_scores.items()
    }
    selected = max(selectors, key=lambda name: (
        validation_results[name]["macro_f1"], -validation_results[name]["activations"],
    ))
    test = collect(host, probes, test_data)
    test_results = {
        name: summarize_policy(test, values, thresholds[name], entropy=name not in selectors)
        for name, values in scores(test).items()
    }
    if fingerprint(host) != original["host_sha256"]:
        raise RuntimeError("Host checkpoint changed")
    report = {
        "seed": original["seed"],
        "host_sha256": original["host_sha256"],
        "acceleration": original["split"]["acceleration"],
        "requested_validation_activation_rate": args.activation_rate,
        "target_validation_activations": count,
        "training_target_counts": dict(zip(("harmful", "neutral", "helpful"),
                                           targets.bincount(minlength=3).tolist(), strict=True)),
        "validation_windows": len(validation.labels),
        "test_windows": len(test.labels),
        "calibrated_entropy_temperature": temperature,
        "validation_selected_benefit_policy": selected,
        "validation": validation_results,
        "test": test_results,
    }
    if args.profile:
        timing = profile_policies(host, probes, test_data, selectors, policies, selected, thresholds, temperature, args.warmup)
        for name in (selected, "io_mf_calibrated"):
            if (timing[name]["accuracy"], timing[name]["gyroscope_activation_rate"]) != (
                test_results[name]["accuracy"], test_results[name]["activation_rate"]
            ):
                raise RuntimeError(f"Live policy disagrees with the cached evaluation for {name}")
        report["live_cpu_profiles"] = timing
        report["live_cpu_warmup"] = args.warmup
    args.output.mkdir(parents=True)
    torch.save({name: model.state_dict() for name, model in selectors.items()}, args.output / "selectors.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {selected} on validation. Wrote {args.output}")


if __name__ == "__main__":
    main()
