from pathlib import Path

import torch
from torch.utils.data import TensorDataset

MODALITIES = ("accelerometer", "gyroscope")
ACCELERATION_STREAMS = {
    "body_acc": ("body_acc",),
    "total_acc": ("total_acc",),
    "body_total_acc": ("body_acc", "total_acc"),
}
GYRO_CHANNELS = 3


def _integers(path):
    return torch.tensor([int(value) for value in path.read_text().split()])


def _signals(folder, split, prefix):
    channels = []
    for axis in "xyz":
        path = folder / "Inertial Signals" / f"{prefix}_{axis}_{split}.txt"
        with path.open() as stream:
            channel = torch.tensor([[float(value) for value in line.split()] for line in stream])
        channels.append(channel)
    return torch.stack(channels, dim=1)


def read_subjects(root, split):
    return _integers(root / split / f"subject_{split}.txt")


def _read(root, split, acceleration):
    folder = root / split
    subjects = read_subjects(root, split)
    labels = _integers(folder / f"y_{split}.txt") - 1
    acc = torch.cat([_signals(folder, split, stream) for stream in ACCELERATION_STREAMS[acceleration]], dim=1)
    gyro = _signals(folder, split, "body_gyro")
    channels = 3 * len(ACCELERATION_STREAMS[acceleration])
    if acc.shape != (len(subjects), channels, 128) or gyro.shape != (len(subjects), GYRO_CHANNELS, 128):
        raise ValueError(f"Unexpected signal shape in {split}")
    if channels != GYRO_CHANNELS:
        gyro = torch.cat((gyro, torch.zeros_like(gyro)), dim=1)
    inputs = torch.stack((acc, gyro), dim=1)
    if len(labels) != len(subjects):
        raise ValueError(f"Unexpected signal shape or row count in {split}")
    if not torch.isfinite(inputs).all() or not ((labels >= 0) & (labels < 6)).all():
        raise ValueError(f"Invalid signals or labels in {split}")
    return inputs, labels, subjects


def load_har(root: Path, validation_subjects, acceleration="body_acc"):
    x, y, subjects = _read(root, "train", acceleration)
    test_x, test_y, test_subjects = _read(root, "test", acceleration)
    official_train = set(subjects.tolist())
    held_out = set(test_subjects.tolist())
    validation = set(validation_subjects)
    if not validation or not validation < official_train or official_train & held_out:
        raise ValueError("Expected disjoint official splits and a proper subset of training subjects for validation")
    mask = torch.isin(subjects, torch.tensor(sorted(validation)))
    train = x[~mask]
    mean = train.mean(dim=(0, 3), keepdim=True)
    scale = train.std(dim=(0, 3), correction=0, keepdim=True)
    scale = torch.where(scale == 0, 1, scale)
    splits = [
        TensorDataset((values - mean) / scale, labels)
        for values, labels in ((train, y[~mask]), (x[mask], y[mask]), (test_x, test_y))
    ]
    manifest = {
        "train_subjects": sorted(official_train - validation),
        "validation_subjects": sorted(validation),
        "test_subjects": sorted(held_out),
        "acceleration": acceleration,
        "mean": mean.flatten().tolist(),
        "scale": scale.flatten().tolist(),
    }
    return (*splits, manifest)
