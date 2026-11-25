"""Compare probe and exact entropy at a validation-selected activation budget."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from src.compare_probes import ARTICLE_THRESHOLD, threshold_for_count
from src.controllers.iomf import distribution, entropy, infer
from src.eval_har import load_run
from src.har import read_subjects
from src.model import WindowInference
from src.run_har import COSTS, OPTIONAL, PRIMARY, metrics


@dataclass
class Windows:
    labels: torch.Tensor
    primary: torch.Tensor
    full: torch.Tensor
    probe: torch.Tensor


@torch.inference_mode()
def collect(host, probes, dataset):
    primary_logits, full_logits, probe_logits = [], [], []
    for window, _ in dataset:
        engine = WindowInference(host, probes, window)
        active = {PRIMARY: engine.encode(PRIMARY)}
        primary_logits.append(engine.logits(active))
        full_logits.append(engine.logits({**active, OPTIONAL: engine.encode(OPTIONAL)}))
        probe_logits.append(engine.logits({**active, OPTIONAL: engine.probe(OPTIONAL)}))
    return Windows(dataset.tensors[1], *(torch.tensor(values, dtype=torch.float64)
                                        for values in (primary_logits, full_logits, probe_logits)))


def calibration_loss(windows, temperature):
    return sum(F.cross_entropy(logits / temperature, windows.labels).item()
               for logits in (windows.primary, windows.full)) / 2


def fit_temperature(validation):
    # Log-spaced fixed search, including the uncalibrated temperature of 1.
    return min((2 ** (index / 32) for index in range(-64, 65)),
               key=lambda value: (calibration_loss(validation, value), abs(value - 1)))


def gains(windows, candidate, temperature):
    def prediction_entropy(logits):
        return entropy(distribution(value / temperature for value in logits))

    return torch.tensor([
        prediction_entropy(primary) - prediction_entropy(optional)
        for primary, optional in zip(windows.primary.tolist(), candidate.tolist(), strict=True)
    ], dtype=torch.float64)


@torch.inference_mode()
def probe_runtime_summary(host, probes, dataset, windows, scores, threshold, temperature):
    active = scores > threshold
    expected = torch.where(active, windows.full.argmax(dim=1), windows.primary.argmax(dim=1))
    for index, (window, _) in enumerate(dataset):
        engine = WindowInference(host, probes, window, temperature=temperature)
        result = infer(engine, (PRIMARY,), (OPTIONAL,), COSTS, threshold)
        if result.prediction != int(expected[index]) or (OPTIONAL in result.active) != bool(active[index]):
            raise RuntimeError(f"Runtime disagrees with cached entropy evaluation at window {index}")
    return {
        **decision_summary(windows, scores, threshold),
        "controller_verified_windows": len(dataset),
    }


def decision_summary(windows, scores, threshold):
    primary = windows.primary.argmax(dim=1)
    full = windows.full.argmax(dim=1)
    activated = scores > threshold
    primary_correct = primary == windows.labels
    full_correct = full == windows.labels
    helpful = ~primary_correct & full_correct
    harmful = primary_correct & ~full_correct
    predictions = torch.where(activated, full, primary)
    return {
        **metrics(windows.labels.tolist(), predictions.tolist()),
        "threshold_nats": threshold,
        "activations": int(activated.sum()),
        "activation_rate": activated.double().mean().item(),
        "corrected": int((activated & helpful).sum()),
        "spoiled": int((activated & harmful).sum()),
        "missed_helpful": int((~activated & helpful).sum()),
        "max_activations_at_nonnegative_threshold": int((scores > 0).sum()),
    }


def host_summary(windows, requested_rate, subjects):
    primary = windows.primary.argmax(dim=1)
    full = windows.full.argmax(dim=1)
    primary_correct = primary == windows.labels
    full_correct = full == windows.labels
    helpful = ~primary_correct & full_correct
    harmful = primary_correct & ~full_correct
    count = len(windows.labels)
    # Diagnostic upper bound: a policy given the true label can spend only on helpful cases.
    best_accuracy = (int(primary_correct.sum()) + min(round(requested_rate * count), int(helpful.sum()))) / count
    return {
        "primary_only": metrics(windows.labels.tolist(), primary.tolist()),
        "early_fusion": metrics(windows.labels.tolist(), full.tolist()),
        "helpful_windows": int(helpful.sum()),
        "harmful_windows": int(harmful.sum()),
        "label_informed_accuracy_upper_bound_at_target": best_accuracy,
        "early_fusion_confusion": torch.bincount(windows.labels * 6 + full, minlength=36).view(6, 6).tolist(),
        "early_fusion_by_subject": [
            {"subject": subject, "windows": int(mask.sum()),
             **metrics(windows.labels[mask].tolist(), full[mask].tolist())}
            for subject in torch.unique(subjects).tolist() if (mask := subjects == subject).any()
        ],
        "early_fusion_per_class": [
            {"class_id": label + 1, "windows": int(mask.sum()),
             "accuracy": full_correct[mask].double().mean().item()}
            for label in range(6) if (mask := windows.labels == label).any()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--activation-rate", type=float, default=0.52)
    args = parser.parse_args()
    if not 0 <= args.activation_rate <= 1 or args.output.exists():
        parser.error("Use an activation rate in [0, 1] and a new output file")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    host, probes, (_, validation_data, test_data), training = load_run(args.dataset, args.run)
    validation = collect(host, probes, validation_data)
    temperature = fit_temperature(validation)
    target = round(args.activation_rate * len(validation.labels))
    policies = {}
    for name, candidate in (("probe", "probe"), ("exact_entropy", "full")):
        for variant, scale in (("uncalibrated", 1.0), ("calibrated", temperature)):
            scores = gains(validation, getattr(validation, candidate), scale)
            threshold = threshold_for_count(scores.tolist(), target)
            policies[f"{name}_{variant}"] = {
                "candidate": candidate,
                "temperature": scale,
                "validation": decision_summary(validation, scores, threshold),
            }

    test = collect(host, probes, test_data)
    for policy in policies.values():
        scale = policy["temperature"]
        scores = gains(test, getattr(test, policy["candidate"]), scale)
        threshold = policy["validation"]["threshold_nats"]
        if policy["candidate"] == "probe":
            policy["test"] = probe_runtime_summary(host, probes, test_data, test, scores, threshold, scale)
        else:
            policy["test"] = decision_summary(test, scores, threshold)

    article_reference = {
        "probe": probe_runtime_summary(host, probes, test_data, test,
                                       gains(test, test.probe, 1.0), ARTICLE_THRESHOLD, 1.0),
        "exact_entropy": decision_summary(test, gains(test, test.full, 1.0), ARTICLE_THRESHOLD),
    }
    all_training_subjects = read_subjects(args.dataset, "train")
    validation_ids = torch.tensor(training["split"]["validation_subjects"])
    validation_subjects = all_training_subjects[torch.isin(all_training_subjects, validation_ids)]
    test_subjects = read_subjects(args.dataset, "test")
    report = {
        "seed": training["seed"],
        "host_sha256": training["host_sha256"],
        "acceleration": training["split"]["acceleration"],
        "inference_configuration": {
            "modality_order": [PRIMARY, OPTIONAL], "primary": [PRIMARY],
            "optional_encoder_cost": COSTS, "fixed_threshold_nats": ARTICLE_THRESHOLD,
            "default_temperature": 1.0,
        },
        "requested_validation_activation_rate": args.activation_rate,
        "target_validation_activations": target,
        "validation_windows": len(validation.labels),
        "test_windows": len(test.labels),
        "calibration": {
            "temperature": temperature,
            "objective": "Mean primary-only and early-fusion cross-entropy on validation",
            "validation_nll_before": calibration_loss(validation, 1.0),
            "validation_nll_after": calibration_loss(validation, temperature),
            "test_nll_before": calibration_loss(test, 1.0),
            "test_nll_after": calibration_loss(test, temperature),
        },
        "host": {
            "validation": host_summary(validation, args.activation_rate, validation_subjects),
            "test": host_summary(test, args.activation_rate, test_subjects),
        },
        "matched_activation": policies,
        "test_at_article_threshold": article_reference,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
