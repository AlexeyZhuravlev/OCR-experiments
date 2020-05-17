from typing import Dict, List

import torch
from torch import nn

from src.data import DataItemKeys
from src.runner import AdditionalDataKeys

from .feature_extraction import FACTORY as extractors_factory
from .prediction_heads import FACTORY as heads_factory
from .prediction_heads import OcrPredictionHead, ATTENTION_HEAD_TYPES
from .modules import SpatialTransformerNetwork

class MultiHeadOcrModel(nn.Module):
    """
    Ocr model with common feature extractor and multiple heads
    """

    def __init__(self, vocab_size: int, input_height: int, input_width: int, stn_params: Dict,\
                 feature_extractor_params: Dict, heads_params: Dict):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        self.spatial_transformer = self._get_spatial_transformer(**stn_params)
        self.feature_extractor = self._get_feature_extractor(**feature_extractor_params)

        height = input_height // self.feature_extractor.vertical_scale
        channels = self.feature_extractor.output_channels

        heads_factory.set_base_params(
            vocab_size=vocab_size,
            input_height=height,
            input_channels=channels
        )

        self.prediction_heads = {}
        self.has_attention_head = False

        for head_key, head_params in heads_params.items():
            self.prediction_heads[head_key] = self._add_head(**head_params)
            super().add_module(head_key, self.prediction_heads[head_key])

    def _get_spatial_transformer(self, use: bool, num_fiducial: int = 20):
        if use:
            shape = (self.input_height, self.input_width)
            return SpatialTransformerNetwork(num_fiducial, shape, shape, 3)
        else:
            return None

    def _get_feature_extractor(self, **kwargs):
        feature_extractor_type = kwargs.pop("type")
        return extractors_factory.get(feature_extractor_type, **kwargs)

    def _add_head(self, **kwargs):
        head_type = kwargs.pop("type")
        head = heads_factory.get(head_type, **kwargs)

        if head_type in ATTENTION_HEAD_TYPES:
            self.has_attention_head = True

        return head

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Gets relevant keys from input data and returns dict with all head outputs
        Returns dict of tensor with predictions with following keys:
        {head_key}.{prediction_key}: torch.Tensor
        """

        images = data[DataItemKeys.IMAGE]
        images_width = data[DataItemKeys.IMAGE_WIDTH]

        if self.spatial_transformer:
            images = self.spatial_transformer(images)

        features = self.feature_extractor(images)
        features_width = images_width // self.feature_extractor.horizontal_scale

        head_input_data = {
            OcrPredictionHead.FEATURES_KEY: features,
            OcrPredictionHead.FEATURES_WIDTH_KEY: features_width
        }

        head_input_data.update(data[AdditionalDataKeys.HEADS_ADDITIONAL_DATA])

        result = {}
        for head_key, head in self.prediction_heads.items():
            output = head(head_input_data)
            for head_output_key, tensor in output.items():
                key = "{}.{}".format(head_key, head_output_key)
                result[key] = tensor

        return result
