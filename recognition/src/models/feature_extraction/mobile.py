import torch
from torch import nn
from .core import BaseFeatureExtractor
from src.models.modules import MBInvertedResidual

class MobileNet(BaseFeatureExtractor):
    def __init__(self, f1=32, f2=64, f3=128, f4=256, f5=512):
        super().__init__()

        self._output_channels = f5

        self.layers = nn.Sequential(
            # Stem convolution
            nn.Conv2d(3, f1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            # Stage 1
            MBInvertedResidual(f1, f1, kernel_size=5, stride=(2, 2), expand_ratio=6),
            MBInvertedResidual(f1, f1, kernel_size=5, expand_ratio=6),
            MBInvertedResidual(f1, f1, kernel_size=5, expand_ratio=6),
            # Stage 2
            MBInvertedResidual(f1, f2, kernel_size=5, stride=(2, 1), expand_ratio=6),
            MBInvertedResidual(f2, f2, kernel_size=5, expand_ratio=6),
            MBInvertedResidual(f2, f2, kernel_size=5, expand_ratio=6),
            # Stage 3
            MBInvertedResidual(f2, f3, kernel_size=5, stride=(2, 2), expand_ratio=6),
            MBInvertedResidual(f3, f3, kernel_size=5, expand_ratio=6),
            MBInvertedResidual(f3, f3, kernel_size=5, expand_ratio=6),
            # Stage 4
            MBInvertedResidual(f3, f4, kernel_size=5, stride=(2, 1), expand_ratio=6),
            MBInvertedResidual(f4, f4, kernel_size=5, expand_ratio=6),
            MBInvertedResidual(f4, f4, kernel_size=5, expand_ratio=6),
            # Stage 5
            MBInvertedResidual(f4, f5, kernel_size=5, stride=(2, 1), expand_ratio=6)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def vertical_scale(self) -> int:
        return 32

    @property
    def horizontal_scale(self) -> int:
        return 4

    @property
    def output_channels(self) -> int:
        return self._output_channels

class CustomMobileNet(BaseFeatureExtractor):
    def __init__(self, f1=32, f2=64, f3=128, f4=192, f5=256):
        super().__init__()

        self._output_channels = f5

        self.layers = nn.Sequential(
            # Stem part
            nn.Conv2d(3, f1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, 3, stride=2, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),

            MBInvertedResidual(f2, f3, kernel_size=5, expand_ratio=6, stride=(2, 2)),

            MBInvertedResidual(f3, f4, kernel_size=5, expand_ratio=3),
            MBInvertedResidual(f4, f4, kernel_size=5, expand_ratio=3, stride=(2, 1)),

            MBInvertedResidual(f4, f5, kernel_size=3, expand_ratio=1),
            MBInvertedResidual(f5, f5, kernel_size=3, expand_ratio=1),
            MBInvertedResidual(f5, f5, kernel_size=3, expand_ratio=1, stride=(2, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def vertical_scale(self) -> int:
        return 16

    @property
    def horizontal_scale(self) -> int:
        return 4

    @property
    def output_channels(self) -> int:
        return self._output_channels
