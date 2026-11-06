"""U-Net with an AlexNet encoder that follows the slide.

Slide:  11x11 -> pool -> 5x5 -> pool -> 3x3 -> 3x3 -> 3x3 -> pool
        channels 96 - 256 - 384 - 384 - 256

Only change: conv1 uses stride 1 instead of 4, so the 256x256 resolution is kept
(stride 4 would leave the decoder with a 64x64 map, too coarse for thin classes like CC).

Input  [B, 1, 256, 256]  ->  logits [B, num_classes, 256, 256]
"""

import torch
from torch import nn


def conv(cin, cout, k):
    # inplace ReLU saves about a third of the activation memory (safe right after a conv)
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2), nn.ReLU(inplace=True))


class AlexNetUNetV2(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = conv(1, 96, 11)       # 256x256x96
        self.conv2 = conv(96, 256, 5)      # 128x128x256
        self.conv3 = conv(256, 384, 3)     #  64x 64x384
        self.conv4 = conv(384, 384, 3)     #  64x 64x384
        self.conv5 = conv(384, 256, 3)     #  64x 64x256
        self.pool = nn.MaxPool2d(3, 2, 1)  # 3x3, stride 2 as in the slide (padding keeps sizes exact)
        self.drop = nn.Dropout2d(dropout)  # AlexNet's dropout idea, placed at the bottleneck

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = conv(256 + 256, 128, 3)  #  64x 64
        self.dec2 = conv(128 + 256, 64, 3)   # 128x128
        self.dec1 = conv(64 + 96, 32, 3)     # 256x256
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.conv1(x)                                       # 256
        s2 = self.conv2(self.pool(s1))                           # 128
        s5 = self.conv5(self.conv4(self.conv3(self.pool(s2))))   #  64
        b = self.drop(self.pool(s5))                             #  32 (bottleneck)

        x = self.dec3(torch.cat([self.up(b), s5], dim=1))
        x = self.dec2(torch.cat([self.up(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up(x), s1], dim=1))
        return self.head(x)