from .simple import SimpleFeatureExtractor
from .resnet import GtcResNet50
from .factory import FeatureExtractorFactory

FACTORY = FeatureExtractorFactory()
FACTORY.register('simple', SimpleFeatureExtractor)
FACTORY.register('resnet50', GtcResNet50)
