"""Train and evaluate paired UCI-HAR runs with a fixed split."""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, stdev
from uuid import uuid4


POLICIES = ("primary_only", "static_fusion", "iomf")
METRICS = ("accuracy", "macro_f1", "mean_latency_us", "p95_latency_us",
           "gyroscope_activation_rate", "mean_committed_modalities", "mean_probes")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, values):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def dataset_hash(root, acceleration):
    digest = hashlib.sha256()
    streams = ("body_acc", "total_acc") if acceleration == "body_total_acc" else (acceleration,)
    for split in ("train", "test"):
        files = [f"subject_{split}.txt", f"y_{split}.txt"]
        files.extend(f"Inertial Signals/{stream}_{axis}_{split}.txt"
                     for stream in (*streams, "body_gyro") for axis in "xyz")
        for relative in files:
            digest.update(f"{split}/{relative}".encode())
            with (root / split / relative).open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def run_module(name, *args):
    subprocess.run([sys.executable, "-m", name, *(str(arg) for arg in args)], check=True)


def verify_training(training, config, seed):
    expected = {
        "seed": seed,
        "host_epochs": config["host_epochs"],
        "probe_epochs": config["probe_epochs"],
    }
    if any(training[key] != value for key, value in expected.items()):
        raise ValueError(f"Training report does not match the configuration for seed {seed}")
    split = training["split"]
    if (split["validation_subjects"], split["acceleration"]) != (
        config["validation_subjects"], config["acceleration"]
    ):
        raise ValueError(f"Training report uses unexpected inputs for seed {seed}")


def statistics(values):
    return {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}


def aggregate(runs):
    policy_stats = {
        policy: {metric: statistics([run["policies"][policy][metric] for run in runs])
                 for metric in METRICS}
        for policy in POLICIES
    }
    paired = {}
    for baseline in ("primary_only", "static_fusion"):
        paired[baseline] = {}
        for metric in METRICS:
            values = [run["policies"]["iomf"][metric] - run["policies"][baseline][metric] for run in runs]
            paired[baseline][metric] = {"per_seed": values, **statistics(values)}
    fidelity = {
        key: statistics([run["probe_fidelity"][key] for run in runs])
        for key in ("threshold_agreement", "mean_absolute_entropy_gain_error_nats")
    }
    return {"policies": policy_stats, "paired_iomf_minus_baseline": paired, "probe_fidelity": fidelity}


def aggregate_activation(reports):
    metrics = ("accuracy", "macro_f1", "activation_rate")
    policies = {}
    for name in reports[0]["matched_activation"]:
        policies[name] = {
            split: {metric: statistics([report["matched_activation"][name][split][metric]
                                       for report in reports]) for metric in metrics}
            for split in ("validation", "test")
        }
    fixed = {
        name: {metric: statistics([report["test_at_article_threshold"][name][metric]
                                  for report in reports]) for metric in metrics}
        for name in reports[0]["test_at_article_threshold"]
    }
    return {"matched_activation": policies, "test_at_article_threshold": fixed}


def train_seed(dataset, output, config, seed):
    destination = output / f"seed-{seed}"
    if destination.exists():
        return
    staging = output / f".seed-{seed}-{uuid4().hex}.training"
    log_path = output / f"seed-{seed}.training.log"
    command = [
        sys.executable, "-m", "src.run_har", str(dataset), str(staging), "--seed", str(seed),
        "--host-epochs", str(config["host_epochs"]), "--probe-epochs", str(config["probe_epochs"]),
        "--acceleration", config["acceleration"], "--validation-subjects",
        *map(str, config["validation_subjects"]),
    ]
    print(f"Training seed {seed}. Log {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_training(read_json(staging / "report.json"), config, seed)
    staging.rename(destination)
    print(f"Training finished for seed {seed}", flush=True)


def aggregate_benefit(reports):
    metrics = ("accuracy", "macro_f1", "activation_rate")
    policy_names = reports[0]["test"]
    policies = {
        name: {metric: statistics([report["test"][name][metric] for report in reports])
               for metric in metrics}
        for name in policy_names
    }
    names = [report["validation_selected_benefit_policy"] for report in reports]
    paired = {}
    for metric in metrics:
        values = [report["test"][name][metric] - report["test"]["io_mf_calibrated"][metric]
                  for report, name in zip(reports, names, strict=True)]
        paired[metric] = {"per_seed": values, **statistics(values)}
    live = {}
    for metric in ("mean_latency_us", "p95_latency_us"):
        live[metric] = {
            key: statistics([report["live_cpu_profiles"][name if key == "selected_benefit" else key][metric]
                             for report, name in zip(reports, names, strict=True)])
            for key in ("selected_benefit", "early_fusion", "io_mf_calibrated")
        }
    return {
        "policies": policies,
        "validation_selected_benefit": {
            "per_seed_policy": names,
            "test": {metric: statistics([report["test"][name][metric]
                                         for report, name in zip(reports, names, strict=True)])
                     for metric in metrics},
        },
        "paired_selected_benefit_minus_calibrated_iomf": paired,
        "live_cpu": live,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--validation-subjects", nargs="+", type=int, default=[1, 3, 5, 6])
    parser.add_argument("--acceleration", choices=["body_acc", "total_acc", "body_total_acc"], default="body_acc")
    parser.add_argument("--host-epochs", type=int, default=100)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--activation-evaluation", nargs="?", const=0.52, type=float,
                        metavar="RATE", help="Evaluate a validation activation budget (default 0.52)")
    parser.add_argument("--benefit-evaluation", nargs="?", const=0.52, type=float,
                        metavar="RATE", help="Evaluate the learned selector (default 0.52 activation)")
    parser.add_argument("--training-workers", type=int, default=1,
                        help="Parallel host training processes; timing evaluations remain serial")
    args = parser.parse_args()
    if args.training_workers < 1:
        parser.error("Use at least one training worker")
    if args.benefit_evaluation is not None and not 0 <= args.benefit_evaluation <= 1:
        parser.error("Use a benefit evaluation rate in [0, 1]")
    if args.activation_evaluation is not None and not 0 <= args.activation_evaluation <= 1:
        parser.error("Use an activation evaluation rate in [0, 1]")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seed identifiers must be unique")
    if not 1 <= args.host_epochs <= 100 or args.probe_epochs < 1 or args.warmup < 0:
        parser.error("Use 1–100 host epochs, positive probe epochs and nonnegative warmup")

    config = {
        "dataset_sha256": dataset_hash(args.dataset, args.acceleration),
        "seeds": args.seeds,
        "validation_subjects": sorted(set(args.validation_subjects)),
        "acceleration": args.acceleration,
        "host_epochs": args.host_epochs,
        "probe_epochs": args.probe_epochs,
        "warmup": args.warmup,
    }
    if args.activation_evaluation is not None:
        config["activation_evaluation_rate"] = args.activation_evaluation
    if args.benefit_evaluation is not None:
        config["benefit_evaluation_rate"] = args.benefit_evaluation
    manifest_path = args.output / "experiment.json"
    if args.output.exists():
        if not manifest_path.is_file() or read_json(manifest_path) != config:
            parser.error("Use a new output directory or resume with the original inputs and options")
    else:
        args.output.mkdir(parents=True)
        write_json(manifest_path, config)

    pending_seeds = [seed for seed in args.seeds if not (args.output / f"seed-{seed}").exists()]
    if args.training_workers > 1 and pending_seeds:
        with ThreadPoolExecutor(max_workers=min(args.training_workers, len(pending_seeds))) as pool:
            futures = [pool.submit(train_seed, args.dataset, args.output, config, seed)
                       for seed in pending_seeds]
            for future in futures:
                future.result()

    runs = []
    benefit_reports = []
    activation_reports = []
    reference_split = None
    for seed in args.seeds:
        destination = args.output / f"seed-{seed}"
        evaluation_path = destination / "evaluation.json"
        print(f"Seed {seed}", flush=True)
        train_seed(args.dataset, args.output, config, seed)

        training = read_json(destination / "report.json")
        verify_training(training, config, seed)
        if reference_split is not None and training["split"] != reference_split:
            raise ValueError(f"Subject split or normalization differs for seed {seed}")
        reference_split = training["split"]
        if not evaluation_path.exists():
            pending = destination / f".evaluation-{uuid4().hex}.json"
            run_module("src.eval_har", args.dataset, destination, pending,
                       "--warmup", args.warmup)
            pending.rename(evaluation_path)
        evaluation = read_json(evaluation_path)
        if (evaluation["seed"], evaluation["host_sha256"], evaluation["measurement"]["warmup"]) != (
            seed, training["host_sha256"], args.warmup
        ):
            raise ValueError(f"Evaluation report does not match training for seed {seed}")
        runs.append(evaluation)
        if args.activation_evaluation is not None:
            activation_path = destination / "activation.json"
            if not activation_path.exists():
                pending = destination / f".activation-{uuid4().hex}.json"
                run_module("src.eval_activation", args.dataset, destination, pending,
                           "--activation-rate", args.activation_evaluation)
                pending.rename(activation_path)
            activation = read_json(activation_path)
            if (activation["seed"], activation["host_sha256"], activation["test_windows"],
                activation["requested_validation_activation_rate"]) != (
                seed, training["host_sha256"], evaluation["test_windows"], args.activation_evaluation
            ):
                raise ValueError(f"Activation report does not match training for seed {seed}")
            activation_reports.append(activation)
        if args.benefit_evaluation is not None:
            benefit_path = destination / "benefit"
            if not benefit_path.exists():
                pending = destination / f".benefit-{uuid4().hex}"
                run_module("src.try_benefit", args.dataset, destination, pending, "--profile",
                           "--warmup", args.warmup, "--activation-rate", args.benefit_evaluation)
                pending.rename(benefit_path)
            benefit = read_json(benefit_path / "report.json")
            if (benefit["seed"], benefit["host_sha256"], benefit["test_windows"],
                benefit["requested_validation_activation_rate"], benefit["live_cpu_warmup"]) != (
                seed, training["host_sha256"], evaluation["test_windows"], args.benefit_evaluation, args.warmup
            ):
                raise ValueError(f"Benefit report does not match training for seed {seed}")
            benefit_reports.append(benefit)
    if any(run["measurement"] != runs[0]["measurement"] or run["test_windows"] != runs[0]["test_windows"]
           for run in runs):
        raise ValueError("Evaluation environment or test window count differs across seeds")

    summary = {
        "configuration": config,
        "test_windows_per_seed": runs[0]["test_windows"],
        "measurement": runs[0]["measurement"],
        "per_seed": [
            {"seed": run["seed"], "report": f"seed-{run['seed']}/evaluation.json",
             "selected_threshold_nats": run["selected_threshold_nats"]}
            for run in runs
        ],
        **aggregate(runs),
    }
    if activation_reports:
        summary["activation_evaluation"] = aggregate_activation(activation_reports)
        for row in summary["per_seed"]:
            row["activation_report"] = f"seed-{row['seed']}/activation.json"
    if benefit_reports:
        summary["benefit_evaluation"] = aggregate_benefit(benefit_reports)
        for row in summary["per_seed"]:
            row["benefit_report"] = f"seed-{row['seed']}/benefit/report.json"
    write_json(args.output / "summary.json", summary)
    print(f"Wrote {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
