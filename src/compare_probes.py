"""Compare embedding MSE and predictive distillation on an existing frozen UCI-HAR host."""

import argparse
import json
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.controllers.iomf import distribution, entropy
from src.eval_har import load_run
from src.har import GYRO_CHANNELS
from src.model import WindowInference
from src.run_har import OPTIONAL, PRIMARY, fingerprint, metrics

ARTICLE_THRESHOLD = 0.06


def distill(host, initial_probe, loader, epochs, objective):
    probe = deepcopy(initial_probe).train()
    projection = probe[-1].requires_grad_(True)
    optimizer = torch.optim.Adam(projection.parameters(), lr=0.001)
    history = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        for inputs, _ in loader:
            signal = inputs[:, 1, :GYRO_CHANNELS]
            mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
            with torch.no_grad():
                primary = host.encoders[PRIMARY](inputs[:, 0])
                full = host.encoders[OPTIONAL](signal)
                teacher = host.fuse(torch.stack((primary, full), dim=1), mask).softmax(dim=1)
            student = host.fuse(torch.stack((primary, probe(signal)), dim=1), mask)
            log_student = student.log_softmax(dim=1)
            if objective == "entropy":
                student_entropy = -(log_student.exp() * log_student).sum(dim=1)
                teacher_entropy = -torch.special.xlogy(teacher, teacher).sum(dim=1)
                loss = F.mse_loss(student_entropy, teacher_entropy)
            else:
                loss = F.kl_div(log_student, teacher, reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(inputs)
        history.append({"epoch": epoch, "training_loss": total / len(loader.dataset)})
        print(f"Probe epoch {epoch}: {objective} loss {history[-1]['training_loss']:.5f}", flush=True)
    return probe.eval(), history


@dataclass(frozen=True)
class WindowScore:
    label: int
    primary_prediction: int
    full_prediction: int
    exact_gain: float
    probe_gains: dict[str, float]


@torch.inference_mode()
def score_windows(host, probes, dataset):
    def predict(engine, embeddings):
        return distribution(engine.logits(embeddings))

    scores = []
    for window, label in dataset:
        engine = WindowInference(host, {}, window)
        active = {PRIMARY: engine.encode(PRIMARY)}
        primary = predict(engine, active)
        full = predict(engine, {**active, OPTIONAL: engine.encode(OPTIONAL)})
        initial = entropy(primary)
        gains = {
            name: initial - entropy(predict(engine, {**active, OPTIONAL: probe(engine.signal(OPTIONAL))}))
            for name, probe in probes.items()
        }
        scores.append(WindowScore(int(label), primary.index(max(primary)), full.index(max(full)),
                                  initial - entropy(full), gains))
    return scores


def summarize(scores, name, threshold):
    labels, predictions = [], []
    activation_count = oracle_misses = false_activations = 0
    error = 0.0
    for row in scores:
        estimated = row.exact_gain if name == "oracle" else row.probe_gains[name]
        select = estimated > threshold
        oracle_select = row.exact_gain > ARTICLE_THRESHOLD
        labels.append(row.label)
        predictions.append(row.full_prediction if select else row.primary_prediction)
        activation_count += select
        oracle_misses += oracle_select and not select
        false_activations += select and not oracle_select
        error += abs(estimated - row.exact_gain)
    count = len(scores)
    return {
        **metrics(labels, predictions),
        "threshold_nats": threshold,
        "gyroscope_activations": activation_count,
        "gyroscope_activation_rate": activation_count / count,
        "agreement_with_oracle_at_article_threshold": 1 - (oracle_misses + false_activations) / count,
        "missed_oracle_activations": oracle_misses,
        "extra_activations": false_activations,
        "mean_absolute_entropy_gain_error_nats": error / count,
    }


def matched_threshold(scores, name, target):
    return threshold_for_count((row.probe_gains[name] for row in scores), target)


def threshold_for_count(scores, target):
    gains = sorted(scores)
    # Runtime uses strict greater-than. Equal scores stay together and thresholds stay nonnegative.
    candidates = {0.0, ARTICLE_THRESHOLD, *(gain for gain in gains if gain >= 0)}
    return min(candidates, key=lambda threshold: (
        abs(len(gains) - bisect_right(gains, threshold) - target),
        len(gains) - bisect_right(gains, threshold) > target,
        abs(threshold - ARTICLE_THRESHOLD),
        threshold,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--objective", choices=["kl", "entropy"], default="kl")
    parser.add_argument("--activation-budget", choices=["incumbent", "oracle"], default="incumbent",
                        help="Match the validation activation count of this policy at 0.06 nats")
    args = parser.parse_args()
    if args.epochs < 1 or args.output.exists():
        parser.error("Use positive epochs and a new output directory")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    host, probes, (train, validation, test), original = load_run(args.dataset, args.run)
    loader = DataLoader(train, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(original["seed"]))
    baseline = probes[OPTIONAL]
    distilled, history = distill(host, baseline, loader, args.epochs, args.objective)
    if fingerprint(host) != original["host_sha256"] or any(p.grad is not None for p in host.parameters()):
        raise RuntimeError("The frozen host changed during distillation")

    candidates = {"embedding_mse": baseline, "distilled": distilled}
    validation_scores = score_windows(host, candidates, validation)
    fixed = {name: summarize(validation_scores, name, ARTICLE_THRESHOLD) for name in candidates}
    reference = fixed["embedding_mse"] if args.activation_budget == "incumbent" else summarize(
        validation_scores, "oracle", ARTICLE_THRESHOLD
    )
    target = reference["gyroscope_activations"]
    matched = {
        name: summarize(validation_scores, name, matched_threshold(validation_scores, name, target))
        for name in candidates
    }
    selected = max(matched, key=lambda name: (
        matched[name]["macro_f1"], -matched[name]["gyroscope_activations"],
    ))
    selected_threshold = matched[selected]["threshold_nats"]

    test_probes = {name: candidates[name] for name in ("embedding_mse", selected)}
    test_scores = score_windows(host, test_probes, test)
    report = {
        "seed": original["seed"], "host_sha256": original["host_sha256"],
        "article_threshold_nats": ARTICLE_THRESHOLD,
        "distillation": {"objective": args.objective, "initialization": "saved embedding-MSE probe",
                         "history": history},
        "activation_budget": {"reference": args.activation_budget, "target_validation_count": target},
        "validation_windows": len(validation_scores), "test_windows": len(test_scores),
        "validation_at_article_threshold": fixed,
        "validation_matched_activation": matched,
        "selection": {"probe": selected, "threshold_nats": selected_threshold,
                      "rule": "Highest validation macro-F1 at the matched activation operating point, "
                              "breaking ties by fewer activations and then the incumbent probe."},
        "test": {
            "embedding_mse_at_article_threshold": summarize(test_scores, "embedding_mse", ARTICLE_THRESHOLD),
            "embedding_mse_matched": summarize(test_scores, "embedding_mse",
                                                matched["embedding_mse"]["threshold_nats"]),
            "selected_policy": summarize(test_scores, selected, selected_threshold),
            "oracle_at_article_threshold": summarize(test_scores, "oracle", ARTICLE_THRESHOLD),
        },
    }

    args.output.mkdir(parents=True)
    torch.save({OPTIONAL: distilled.state_dict()}, args.output / "distilled_probes.pt")
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {selected} at {selected_threshold:.6f} nats. Wrote {args.output}")


if __name__ == "__main__":
    main()
