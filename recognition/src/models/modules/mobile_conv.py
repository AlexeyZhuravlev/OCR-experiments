from collections import OrderedDict
from torch import nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=(1, 1), groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class MBInvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=(1, 1), expand_ratio=6):
        super(MBInvertedResidual, self).__init__()

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == (1, 1) or stride == 1) and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
