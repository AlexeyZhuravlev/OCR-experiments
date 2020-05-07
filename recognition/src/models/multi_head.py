import torch
from torch import nn

from typing import Dict, List

from .feature_extraction import FACTORY as extractors_factory
from .prediction_heads import FACTORY as heads_factory


class MultiHeadOcrModel(nn.Module):

    def __init__(self, vocab_path: str, input_height: int, feature_extractor_name: str,\
                 feature_extractor_params: Dict, heads_params: Dict):
        super().__init__()

        self.feature_extractor = extractors_factory.get(feature_extractor_name,\
                                                        feature_extractor_params)

        height = input_height // self.feature_extractor.vertical_scale
        channels = self.feature_extractor.output_channels
        heads_common_params = {
            'vocab_path': vocab_path,
            'input_height': height,
            'input_channels': channels
        }
        heads_factory.set_base_params(heads_common_params)

        self.prediction_heads = {}

        for head_key, head_params in heads_params.items():
            head_type = head_params["type"]
            head_specific_params = head_params["specific_params"]
            self.prediction_heads[head_key] = heads_factory.get(head_type, head_specific_params)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)

        result = {}
        for key, head in self.prediction_heads.items():
            output = head(features)
            result[key] = output

        return result
