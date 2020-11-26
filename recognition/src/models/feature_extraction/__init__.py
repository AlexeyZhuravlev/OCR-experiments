from .resnet14 import SimpleResNet14
from .resnet29 import ResNet29
from .resnet40 import GtcResNet40
from .mobile import MobileNet
from .factory import FeatureExtractorFactory

FACTORY = FeatureExtractorFactory()
FACTORY.register('resnet14', SimpleResNet14)
FACTORY.register('resnet29', ResNet29)
FACTORY.register('resnet40', GtcResNet40)
FACTORY.register('mobile', MobileNet)
