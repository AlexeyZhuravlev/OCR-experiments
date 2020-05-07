from .simple import SimpleFeatureExtractor
from .factory import FeatureExtractorFactory

FACTORY = FeatureExtractorFactory()
FACTORY.register('simple', SimpleFeatureExtractor)
