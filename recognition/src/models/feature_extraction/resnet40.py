"""
    Modified resnet-50 architecture from
    https://aaai.org/Papers/AAAI/2020GB/AAAI-HuW.7838.pdf
"""

import torch
from torch import nn
from .core import BaseFeatureExtractor
from torchvision.models.resnet import Bottleneck, conv1x1

def get_downsample(channels_in, channels_out, stride=1):
    """Standart resnet normalziation. Same as in torchvision.resnet"""
    return nn.Sequential(
        conv1x1(channels_in, channels_out, stride),
        nn.BatchNorm2d(channels_out)
    )

class GtcResNet40(BaseFeatureExtractor):
    def __init__(self, c0=64, c1=64, c2=128, c3=256, c4=512):
        super().__init__()
        self._output_channels = c4 * 4

        self.layers = nn.Sequential(
            # Root network: conv-bn-relu-maxpool
            nn.Conv2d(3, c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # First 3 blocks
            Bottleneck(c0, c1, norm_layer=nn.BatchNorm2d,
                       downsample=get_downsample(c0, c1 * 4)),
            Bottleneck(c1 * 4, c1, norm_layer=nn.BatchNorm2d),
            Bottleneck(c1 * 4, c1, norm_layer=nn.BatchNorm2d),
            # Second 4 blocks
            Bottleneck(c1 * 4, c2, norm_layer=nn.BatchNorm2d,
                       downsample=get_downsample(c1 * 4, c2 * 4)),
            Bottleneck(c2 * 4, c2, norm_layer=nn.BatchNorm2d),
            Bottleneck(c2 * 4, c2, norm_layer=nn.BatchNorm2d),
            Bottleneck(c2 * 4, c2, norm_layer=nn.BatchNorm2d),
            # Height downsampling
            nn.MaxPool2d((2, 1), (2, 1)),
            # Third 3 blocks
            Bottleneck(c2 * 4, c3, norm_layer=nn.BatchNorm2d,
                       downsample=get_downsample(c2 * 4, c3 * 4)),
            Bottleneck(c3 * 4, c3, norm_layer=nn.BatchNorm2d),
            Bottleneck(c3 * 4, c3, norm_layer=nn.BatchNorm2d),
            # Height downsampling
            nn.MaxPool2d((2, 1), (2, 1)),
            # Last 3 blocks
            Bottleneck(c3 * 4, c4, norm_layer=nn.BatchNorm2d,
                       downsample=get_downsample(c3* 4, c4 * 4)),
            Bottleneck(c4 * 4, c4, norm_layer=nn.BatchNorm2d),
            Bottleneck(c4 * 4, c4, norm_layer=nn.BatchNorm2d)
        )

    @property
    def vertical_scale(self) -> int:
        return 16

    @property
    def horizontal_scale(self) -> int:
        return 4

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
