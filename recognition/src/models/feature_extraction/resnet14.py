"""
    Light resnet-block based architecture of 14 total conv layers
"""

import torch
from torch import nn
from .core import BaseFeatureExtractor
from torchvision.models import resnet

def get_downsample(channels_in, channels_out, stride=1):
    """Standart resnet normalziation. Same as in torchvision.resnet"""
    return nn.Sequential(
        resnet.conv1x1(channels_in, channels_out, stride),
        nn.BatchNorm2d(channels_out)
    )

class SimpleResNet14(BaseFeatureExtractor):
    """
    Simple OCR model by-design inspired by resnet-18 classification network
    """
    def __init__(self, initial_pool=4):
        """
        initial_pool - pooling degree after first two convs
        4 is suggested for height=64, 2 for height=32
        """
        super().__init__()
        self._initial_pool = initial_pool
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(initial_pool, initial_pool),
            resnet.BasicBlock(64, 128, downsample=get_downsample(64, 128),
                              norm_layer=nn.BatchNorm2d),
            resnet.BasicBlock(128, 128, norm_layer=nn.BatchNorm2d),
            nn.MaxPool2d(2, 2),
            resnet.BasicBlock(128, 192, downsample=get_downsample(128, 192),
                              norm_layer=nn.BatchNorm2d),
            resnet.BasicBlock(192, 192, norm_layer=nn.BatchNorm2d),
            nn.MaxPool2d(2, 2),
            resnet.BasicBlock(192, 256, downsample=get_downsample(192, 256),
                              norm_layer=nn.BatchNorm2d),
            resnet.BasicBlock(256, 256, norm_layer=nn.BatchNorm2d),
            nn.MaxPool2d((2, 1), (2, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def vertical_scale(self) -> int:
        return 8 * self._initial_pool

    @property
    def horizontal_scale(self) -> int:
        return 4 * self._initial_pool

    @property
    def output_channels(self) -> int:
        return 256
