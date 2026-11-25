import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.controllers.iomf import Timing, distribution, infer
from src.har import ACCELERATION_STREAMS, GYRO_CHANNELS, MODALITIES, load_har
from src.model import Host, Probe, WindowInference

PRIMARY, OPTIONAL = MODALITIES
COSTS = {OPTIONAL: 1.0}


def metrics(labels, predictions):
    scores = []
    correct = sum(label == prediction for label, prediction in zip(labels, predictions, strict=True))
    for category in range(6):
        tp = sum(label == prediction == category for label, prediction in zip(labels, predictions))
        denominator = labels.count(category) + predictions.count(category)
        scores.append(2 * tp / denominator if denominator else 0.0)
    return {"accuracy": correct / len(labels), "macro_f1": sum(scores) / len(scores)}


def train_host(train, validation, max_epochs):
    host = Host(acceleration_channels=train.dataset.tensors[0].shape[2])
    optimizer = torch.optim.Adam(host.parameters(), lr=0.001, weight_decay=0.0001)
    best_score, best_epoch = -1, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        host.train()
        training_loss = 0.0
        for inputs, labels in train:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(host(inputs), labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(labels)
        host.eval()
        validation_loss = 0.0
        labels, predictions = [], []
        with torch.inference_mode():
            for inputs, targets in validation:
                logits = host(inputs)
                validation_loss += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
                predictions.extend(logits.argmax(1).tolist())
                labels.extend(targets.tolist())
        validation_metrics = metrics(labels, predictions)
        score = validation_metrics["macro_f1"]
        row = {
            "epoch": epoch, "training_loss": training_loss / len(train.dataset),
            "validation_loss": validation_loss / len(labels), **validation_metrics,
        }
        history.append(row)
        print("Host", json.dumps(row), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = deepcopy(host.state_dict())
        if epoch - best_epoch >= 10:
            break
    host.load_state_dict(best_state)
    return host.requires_grad_(False).eval(), best_epoch, history


def fingerprint(host):
    digest = hashlib.sha256()
    for name, tensor in host.state_dict().items():
        digest.update(name.encode())
        digest.update(bytes(tensor.contiguous().view(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def fit_probe(host, train, epochs):
    encoder = host.encoders[OPTIONAL]
    probe = Probe(encoder).train()
    optimizer = torch.optim.Adam(probe[-1].parameters(), lr=0.001)
    for _ in range(epochs):
        for inputs, _ in train:
            signal = inputs[:, 1, :GYRO_CHANNELS]
            with torch.no_grad():
                target = encoder(signal)
            optimizer.zero_grad()
            nn.functional.mse_loss(probe(signal), target).backward()
            optimizer.step()
    return {OPTIONAL: probe.eval()}


@torch.inference_mode()
def evaluate(host, probes, dataset, threshold, *, deadline=None, timing=None, baselines=False):
    labels, predictions, records = [], [], []
    primary_predictions, static_predictions = [], []
    for window, label in dataset:
        engine = WindowInference(host, probes, window)
        result = infer(engine, (PRIMARY,), (OPTIONAL,), COSTS, threshold,
                       deadline=deadline, timing=timing)
        labels.append(int(label))
        predictions.append(result.prediction)
        records.append({**asdict(result), "label": int(label), "prediction": result.prediction})
        if baselines:
            primary = {PRIMARY: engine.encode(PRIMARY)}
            logits = engine.logits(primary)
            primary_predictions.append(max(range(len(logits)), key=logits.__getitem__))
            probabilities = distribution(engine.logits({**primary, OPTIONAL: engine.encode(OPTIONAL)}))
            static_predictions.append(max(range(len(probabilities)), key=probabilities.__getitem__))
    summary = metrics(labels, predictions)
    summary["infeasible"] = predictions.count(None)
    summary["optional_activation_rate"] = sum(OPTIONAL in row["active"] for row in records) / len(records)
    if timing is not None:
        accounted = [deadline - row["residual"] for row in records if row["prediction"] is not None]
        summary["mean_accounted_time_us"] = sum(accounted) / len(accounted) if accounted else None
        summary["primary_infeasibility_rate"] = summary["infeasible"] / len(records)
        summary["modeled_timely_accuracy"] = summary["accuracy"]
    if baselines:
        summary["primary_only"] = metrics(labels, primary_predictions)
        summary["static_fusion"] = metrics(labels, static_predictions)
    return summary, records


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an IO-MF UCI-HAR reference run")
    parser.add_argument("dataset", type=Path, help="Extracted UCI HAR Dataset folder")
    parser.add_argument("output", type=Path, help="New directory for run artifacts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-subjects", nargs="+", type=int, default=[1, 3, 5, 6])
    parser.add_argument("--acceleration", choices=ACCELERATION_STREAMS, default="body_acc")
    parser.add_argument("--host-epochs", type=int, choices=range(1, 101), default=100, metavar="1..100")
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--timing", type=Path, help="JSON with deadline, primary, probe and commit bounds in microseconds")
    args = parser.parse_args()
    if args.probe_epochs < 1 or args.output.exists():
        parser.error("Use at least one probe epoch and a new output directory")
    timing_config = json.loads(args.timing.read_text(encoding="utf-8-sig")) if args.timing else None
    timing = Timing(timing_config["primary"], timing_config["probe"], timing_config["commit"]) if timing_config else None
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train, validation, test, manifest = load_har(args.dataset, args.validation_subjects, args.acceleration)
    train_loader = DataLoader(train, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    host, best_epoch, history = train_host(train_loader, DataLoader(validation, batch_size=64), args.host_epochs)
    host_hash = fingerprint(host)
    probes = fit_probe(host, train_loader, args.probe_epochs)
    sweep = []
    for index in range(16):
        threshold = index / 50
        summary, _ = evaluate(host, probes, validation, threshold)
        sweep.append({"threshold_nats": threshold, **summary})
    selected = max(sweep, key=lambda row: (row["macro_f1"], -row["optional_activation_rate"]))
    threshold = selected["threshold_nats"]
    summary, records = evaluate(host, probes, test, threshold, baselines=True)
    deadline_result = None
    if timing:
        deadline_result = evaluate(host, probes, test, threshold, deadline=timing_config["deadline"], timing=timing)
    if fingerprint(host) != host_hash:
        raise RuntimeError("Frozen host changed during probe fitting or evaluation")
    args.output.mkdir(parents=True)
    torch.save(host.state_dict(), args.output / "host.pt")
    torch.save({name: probe.state_dict() for name, probe in probes.items()}, args.output / "probes.pt")
    report = {
        "seed": args.seed, "torch_version": torch.__version__, "host_epochs": args.host_epochs,
        "selected_epoch": best_epoch, "host_history": history, "probe_epochs": args.probe_epochs, "split": manifest,
        "host_sha256": host_hash, "selected_threshold_nats": threshold, "validation_sweep": sweep,
        "test_iomf": summary, "timing_input_us": timing_config,
        "inference_configuration": {
            "modality_order": list(MODALITIES), "primary": [PRIMARY],
            "optional_encoder_cost": COSTS, "temperature": 1.0, "threshold_nats": threshold,
        },
    }
    if deadline_result:
        report["test_iomf_d"] = deadline_result[0]
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, rows in [("iomf", records), ("iomf_d", deadline_result[1] if deadline_result else [])]:
        if rows:
            with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
