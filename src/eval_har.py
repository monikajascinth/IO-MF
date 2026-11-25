"""Evaluate frozen UCI-HAR checkpoints and optional modeled deadline policies."""

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

import torch

from src.controllers.iomf import Timing, distribution, entropy, infer
from src.har import load_har
from src.model import Host, Probe, WindowInference
from src.run_har import COSTS, OPTIONAL, PRIMARY, fingerprint, metrics


@dataclass
class Observation:
    prediction: int | None
    elapsed_ns: int
    active_count: int
    probe_count: int


def exact_policy(engine, *, include_optional):
    active = {PRIMARY: engine.encode(PRIMARY)}
    if include_optional:
        active[OPTIONAL] = engine.encode(OPTIONAL)
    probabilities = distribution(engine.logits(active))
    return max(range(len(probabilities)), key=probabilities.__getitem__), len(active), 0


def controller_policy(engine, threshold, *, deadline=None, timing=None):
    result = infer(engine, (PRIMARY,), (OPTIONAL,), COSTS, threshold,
                   deadline=deadline, timing=timing)
    return result.prediction, len(result.active), sum(event.action == "probe" for event in result.trace)


def profile(host, probes, dataset, policy, warmup, *, temperature=1.0):
    for index in range(warmup):
        window, _ = dataset[index % len(dataset)]
        policy(WindowInference(host, probes, window, temperature=temperature))

    observations = []
    for window, _ in dataset:
        engine = WindowInference(host, probes, window, temperature=temperature)
        start = perf_counter_ns()
        prediction, active, probes_run = policy(engine)
        elapsed = perf_counter_ns() - start
        observations.append(Observation(prediction, elapsed, active, probes_run))
    return observations


def summarize(labels, observations, deadline_us=None):
    predictions = [row.prediction for row in observations]
    times = sorted(row.elapsed_ns for row in observations)
    count = len(times)
    summary = {
        **metrics(labels, predictions),
        "mean_latency_us": mean(times) / 1000,
        "p95_latency_us": times[(95 * count + 99) // 100 - 1] / 1000,
        "mean_committed_modalities": mean(row.active_count for row in observations),
        "mean_probes": mean(row.probe_count for row in observations),
        "gyroscope_activation_rate": sum(row.active_count == 2 for row in observations) / count,
        "primary_infeasibility_rate": predictions.count(None) / count,
    }
    if deadline_us is not None:
        bound = deadline_us * 1000
        summary["observed_deadline_violation_rate"] = sum(t > bound for t in times) / count
        summary["observed_timely_accuracy"] = sum(
            label == row.prediction and row.elapsed_ns <= bound
            for label, row in zip(labels, observations, strict=True)
        ) / count
    return summary


def probe_diagnostic(host, probes, test, threshold):
    labels, oracle_predictions = [], []
    errors, agreements, oracle_activations = [], [], []
    probe_only, oracle_only = 0, 0
    corrected, degraded = 0, 0
    stage_times = {"primary": [], "probe": [], "commit": []}
    penalty = threshold * COSTS[OPTIONAL]
    for window, label in test:
        engine = WindowInference(host, probes, window)
        start = perf_counter_ns()
        active = {PRIMARY: engine.encode(PRIMARY)}
        primary = distribution(engine.logits(active))
        initial_entropy = entropy(primary)
        stage_times["primary"].append(perf_counter_ns() - start)

        start = perf_counter_ns()
        surrogate = distribution(engine.logits({**active, OPTIONAL: engine.probe(OPTIONAL)}))
        estimate = initial_entropy - entropy(surrogate)
        stage_times["probe"].append(perf_counter_ns() - start)

        start = perf_counter_ns()
        full = distribution(engine.logits({**active, OPTIONAL: engine.encode(OPTIONAL)}))
        exact = initial_entropy - entropy(full)
        stage_times["commit"].append(perf_counter_ns() - start)

        probe_activates = estimate - penalty > 0
        oracle_activates = exact - penalty > 0
        errors.append(abs(estimate - exact))
        agreements.append(probe_activates == oracle_activates)
        oracle_activations.append(oracle_activates)
        probe_only += probe_activates and not oracle_activates
        oracle_only += oracle_activates and not probe_activates
        label = int(label)
        labels.append(label)
        primary_prediction = max(range(len(primary)), key=primary.__getitem__)
        full_prediction = max(range(len(full)), key=full.__getitem__)
        oracle_predictions.append(full_prediction if oracle_activates else primary_prediction)
        if oracle_activates:
            corrected += primary_prediction != label and full_prediction == label
            degraded += primary_prediction == label and full_prediction != label

    fidelity = {
        "threshold_agreement": mean(agreements),
        "mean_absolute_entropy_gain_error_nats": mean(errors),
        "candidate_ranking_agreement": None,
        "candidate_count": 1,
        "probe_only_activations": probe_only,
        "oracle_only_activations": oracle_only,
    }
    oracle = {
        **metrics(labels, oracle_predictions),
        "gyroscope_activation_rate": mean(oracle_activations),
        "corrected_primary_predictions": corrected,
        "degraded_primary_predictions": degraded,
    }
    timing = {
        "scope": "Diagnostic per-stage measurements with all stages executed on every window. "
                 "Each stage includes its fusion, head, and entropy work. Use the separate policy "
                 "profiles for end-to-end inference latency.",
        "mean_us": {name: mean(values) / 1000 for name, values in stage_times.items()},
    }
    return fidelity, oracle, timing


def load_run(dataset, run):
    training = json.loads((run / "report.json").read_text(encoding="utf-8"))
    split = training["split"]
    train, validation, test, manifest = load_har(dataset, split["validation_subjects"], split["acceleration"])
    if manifest != split:
        raise ValueError("Dataset split or normalization differs from the saved run")
    host = Host(acceleration_channels=train.tensors[0].shape[2]).eval()
    host.load_state_dict(torch.load(run / "host.pt", map_location="cpu", weights_only=True))
    host.requires_grad_(False)
    if fingerprint(host) != training["host_sha256"]:
        raise ValueError("Host checkpoint differs from the recorded fingerprint")
    states = torch.load(run / "probes.pt", map_location="cpu", weights_only=True)
    probes = {name: Probe(host.encoders[name]).eval().requires_grad_(False) for name in states}
    for name, probe in probes.items():
        probe.load_state_dict(states[name])
    return host, probes, (train, validation, test), training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run", type=Path, help="Training run with report and checkpoints")
    parser.add_argument("output", type=Path, help="New JSON report file")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--timing", type=Path, help="JSON with primary, probe and commit bounds in microseconds")
    parser.add_argument("--deadlines-us", type=int, nargs="+")
    args = parser.parse_args()
    if args.warmup < 0 or args.output.exists():
        parser.error("Use a nonnegative warmup count and a new output filename")
    if bool(args.timing) != bool(args.deadlines_us):
        parser.error("Supply --timing and --deadlines-us together")
    if args.deadlines_us and any(value <= 0 for value in args.deadlines_us):
        parser.error("Deadlines must be positive")
    timing = None
    if args.timing:
        bounds = json.loads(args.timing.read_text(encoding="utf-8-sig"))
        timing = Timing(bounds["primary"], bounds["probe"], bounds["commit"])

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    host, probes, (_, _, test), training = load_run(args.dataset, args.run)
    threshold = training["selected_threshold_nats"]
    labels = test.tensors[1].tolist()
    with torch.inference_mode():
        policies = {
            "primary_only": lambda engine: exact_policy(engine, include_optional=False),
            "static_fusion": lambda engine: exact_policy(engine, include_optional=True),
            "iomf": lambda engine: controller_policy(engine, threshold),
        }
        observed = {name: profile(host, probes, test, policy, args.warmup)
                    for name, policy in policies.items()}
        fidelity, oracle, stages = probe_diagnostic(host, probes, test, threshold)
        report = {
            "seed": training["seed"],
            "host_sha256": training["host_sha256"],
            "selected_threshold_nats": threshold,
            "test_windows": len(labels),
            "measurement": {
                "device": "cpu", "platform": platform.platform(), "python": platform.python_version(),
                "torch": torch.__version__, "threads": 1, "batch_size": 1, "warmup": args.warmup,
                "scope": "Per-window inference on prepared inputs, including encoders, probes, fusion, "
                         "head and controller. Disk loading and normalization are outside the timer.",
            },
            "probe_fidelity": fidelity,
            "exact_entropy_oracle": oracle,
            "diagnostic_stage_timing": stages,
            "policies": {name: summarize(labels, rows) for name, rows in observed.items()},
        }
        if timing:
            report["modeled_operation_bounds_us"] = bounds
            report["deadline_sweep"] = []
            for deadline in sorted(set(args.deadlines_us)):
                def deadline_policy(engine):
                    return controller_policy(engine, threshold, deadline=deadline, timing=timing)

                deadline_rows = profile(host, probes, test, deadline_policy, args.warmup)
                report["deadline_sweep"].append({
                    "deadline_us": deadline,
                    "iomf_d": summarize(labels, deadline_rows, deadline),
                    **{name: summarize(labels, rows, deadline) for name, rows in observed.items()},
                })
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Evaluated {len(labels)} windows. Wrote {args.output}")


if __name__ == "__main__":
    main()
