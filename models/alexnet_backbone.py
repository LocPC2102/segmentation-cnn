from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class BackboneFeatures(NamedTuple):
    stage1: torch.Tensor      # [batch,  64, 256, 256]
    stage2: torch.Tensor      # [batch, 192, 128, 128]
    stage3: torch.Tensor      # [batch, 384,  64,  64]
    stage4: torch.Tensor      # [batch, 256,  32,  32]
    bottleneck: torch.Tensor  # [batch, 256,  16,  16]


class AlexNetBackbone(nn.Module):
    """AlexNet-style encoder (channels 64-192-384-256-256) for 256x256 grayscale slices."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, images: torch.Tensor) -> BackboneFeatures:
        stage1 = self.relu(self.conv1(images))
        stage2 = self.relu(self.conv2(self.pool(stage1)))
        stage3 = self.relu(self.conv3(self.pool(stage2)))
        stage4 = self.relu(self.conv4(self.pool(stage3)))
        bottleneck = self.relu(self.conv5(self.pool(stage4)))
        return BackboneFeatures(stage1, stage2, stage3, stage4, bottleneck)