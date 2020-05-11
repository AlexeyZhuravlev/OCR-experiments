from typing import Dict, List

import torch
from torch import nn

from src.data import DataItemKeys

from .feature_extraction import FACTORY as extractors_factory
from .prediction_heads import FACTORY as heads_factory
from .prediction_heads import OcrPredictionHead


class MultiHeadOcrModel(nn.Module):
    """
    Ocr model with common feature extractor and multiple heads
    """

    def __init__(self, vocab_size: int, input_height: int, feature_extractor_name: str,\
                 feature_extractor_params: Dict, heads_params: Dict):
        super().__init__()

        self.feature_extractor = extractors_factory.get(feature_extractor_name,\
                                                        feature_extractor_params)

        height = input_height // self.feature_extractor.vertical_scale
        channels = self.feature_extractor.output_channels
        heads_common_params = {
            "vocab_size": vocab_size,
            "input_height": height,
            "input_channels": channels
        }
        heads_factory.set_base_params(heads_common_params)

        self.prediction_heads = {}

        for head_key, head_params in heads_params.items():
            head_type = head_params["type"]
            head_specific_params = head_params["specific_params"]
            self.prediction_heads[head_key] = heads_factory.get(head_type, head_specific_params)
            super().add_module(head_key, self.prediction_heads[head_key])

    def forward(self, data : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Gets relevant keys from input data and returns dict with all head outputs
        Returns dict of tensor with predictions with following keys:
        {head_key}.{prediction_key}: torch.Tensor
        """

        images = data[DataItemKeys.IMAGE]
        images_width = data[DataItemKeys.IMAGE_WIDTH]

        features = self.feature_extractor(images)
        features_width = images_width // self.feature_extractor.horizontal_scale

        head_input_data = {
            OcrPredictionHead.FEATURES_KEY: features,
            OcrPredictionHead.FEATURES_WIDTH_KEY: features_width
        }

        result = {}
        for head_key, head in self.prediction_heads.items():
            output = head(head_input_data)
            for head_output_key, tensor in output.items():
                key = "{}.{}".format(head_key, head_output_key)
                result[key] = tensor

        return result
