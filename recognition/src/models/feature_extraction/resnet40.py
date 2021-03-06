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
    def __init__(self, initial_stride=2, c0=64, c1=64, c2=128, c3=256, c4=512):
        """
        initial_stride - stride applied at the first convolution,
        might be useful to change alongside with input height;
        Original paper proposes value 2 for input height 64
        """
        super().__init__()
        self._output_channels = c4 * 4
        self._initial_stride = initial_stride

        self.layers = nn.Sequential(
            # Root network: conv-bn-relu-maxpool
            nn.Conv2d(3, c0, kernel_size=7, stride=initial_stride, padding=3, bias=False),
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def vertical_scale(self) -> int:
        return self._initial_stride * 8

    @property
    def horizontal_scale(self) -> int:
        return self._initial_stride * 2

    @property
    def output_channels(self) -> int:
        return self._output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.layers(x)
        return result
