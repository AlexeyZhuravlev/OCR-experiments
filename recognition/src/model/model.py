from torch import nn
from feature_extraction import SimpleFeatureExtractor

class OcrModel(nn.Module):
    def __init__(self):
        self.feature_extractor = SimpleFeatureExtractor()
        