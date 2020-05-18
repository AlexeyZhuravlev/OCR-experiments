"""
    Light resnet-block based architecture of 14 total conv layers
"""

import torch
from torch import nn
from .core import BaseFeatureExtractor
from torchvision.models import resnet

class SimpleResNet14(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            resnet.BasicBlock(64, 128, downsample=nn.Conv2d(64, 128, 1, 1, 0)),
            resnet.BasicBlock(128, 128),
            nn.MaxPool2d(2, 2),
            resnet.BasicBlock(128, 192, downsample=nn.Conv2d(128, 192, 1, 1, 0)),
            resnet.BasicBlock(192, 192),
            nn.MaxPool2d(2, 2),
            resnet.BasicBlock(192, 256, downsample=nn.Conv2d(192, 256, 1, 1, 0)),
            resnet.BasicBlock(256, 256),
            nn.MaxPool2d((2, 1), (2, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def vertical_scale(self) -> int:
        return 32

    @property
    def horizontal_scale(self) -> int:
        return 16

    @property
    def output_channels(self) -> int:
        return 256
