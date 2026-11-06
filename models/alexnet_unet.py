from __future__ import annotations

import torch
from torch import nn

from src.cnn_segmentation.models.alexnet_backbone import AlexNetBackbone


class AlexNetUNet(nn.Module):
    """U-Net with an AlexNet-style backbone. Input [B,1,256,256] -> logits [B,C,256,256]."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = AlexNetBackbone()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec4 = nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1)  # 32x32
        self.dec3 = nn.Conv2d(128 + 384, 64, kernel_size=3, padding=1)   # 64x64
        self.dec2 = nn.Conv2d(64 + 192, 32, kernel_size=3, padding=1)    # 128x128
        self.dec1 = nn.Conv2d(32 + 64, 16, kernel_size=3, padding=1)     # 256x256
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def _merge(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.upsample(features), skip], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        f = self.backbone(images)
        x = self.relu(self.dec4(self._merge(f.bottleneck, f.stage4)))
        x = self.relu(self.dec3(self._merge(x, f.stage3)))
        x = self.relu(self.dec2(self._merge(x, f.stage2)))
        x = self.relu(self.dec1(self._merge(x, f.stage1)))
        return self.head(x)