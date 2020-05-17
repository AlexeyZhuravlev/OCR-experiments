from typing import Dict
from .core import BaseFeatureExtractor

class FeatureExtractorFactory:
    def __init__(self):
        self.feature_extractors = {}

    def register(self, name: str, extractor):
        self.feature_extractors[name] = extractor

    def get(self, name: str, **kwargs) -> BaseFeatureExtractor:
        extractor = self.feature_extractors.get(name)
        if not extractor:
            raise ValueError(name)
        return extractor(**kwargs)
