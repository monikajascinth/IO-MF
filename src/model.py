from copy import deepcopy
from math import isfinite

import torch
from torch import nn

from src.har import GYRO_CHANNELS, MODALITIES


def convolution(in_channels, out_channels, kernel):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class Host(nn.Module):
    def __init__(self, acceleration_channels=3):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                convolution(acceleration_channels if name == MODALITIES[0] else GYRO_CHANNELS, 32, 5),
                convolution(32, 64, 5),
                convolution(64, 128, 3),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
            )
            for name in MODALITIES
        })
        self.head = nn.Sequential(nn.Dropout(0.30), nn.Linear(258, 6))

    def fuse(self, embeddings, mask):
        masked = embeddings.masked_fill(~mask.unsqueeze(-1), 0)
        return self.head(torch.cat((masked.flatten(1), mask.to(embeddings.dtype)), dim=1))

    def forward(self, inputs):
        embeddings = torch.stack((
            self.encoders[MODALITIES[0]](inputs[:, 0]),
            self.encoders[MODALITIES[1]](inputs[:, 1, :GYRO_CHANNELS]),
        ), dim=1)
        mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        if self.training:
            mask[:, 1:] = torch.rand_like(mask[:, 1:], dtype=torch.float32) >= 0.20
        return self.fuse(embeddings, mask)


class Probe(nn.Sequential):
    def __init__(self, encoder):
        super().__init__(
            deepcopy(encoder[0]).requires_grad_(False),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(32, 128),
        )

    def train(self, mode=True):
        super().train(mode)
        self[0].eval()
        return self


class WindowInference:
    def __init__(self, host, probes, window, *, temperature=1.0):
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.temperature = temperature
        self.host = host
        self.probes = probes
        self.window = window

    def signal(self, modality):
        signal = self.window[None, MODALITIES.index(modality)]
        return signal if modality == MODALITIES[0] else signal[:, :GYRO_CHANNELS]

    def encode(self, modality):
        return self.host.encoders[modality](self.signal(modality))

    def probe(self, modality):
        return self.probes[modality](self.signal(modality))

    def logits(self, embeddings):
        zero = self.window.new_zeros((1, 128))
        slots = torch.stack([embeddings.get(name, zero) for name in MODALITIES], dim=1)
        mask = torch.tensor([[name in embeddings for name in MODALITIES]], device=self.window.device)
        logits = self.host.fuse(slots, mask)[0].tolist()
        return [value / self.temperature for value in logits]
