from torchvision.models import resnet
from torch import nn

class SimpleFeatureExtractor(nn.Sequential):
    def __init__(self):
        super(SimpleFeatureExtractor, self).__init__(
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

