import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict

class HeadOutputKeys:
    """
    Keys, used in models output dictionaries with valid information for loss calculation
    """
    LOG_PROBS = "logprobs"
    LOG_PROBS_LEN = "logprobs_len"
    LOGITS = "logits"

class OcrPredictionHead(ABC, nn.Module):

    # Keys of dict which heads might expect in their input
    # Tensor of shape (B, C, H, W) of extracted conv features
    FEATURES_KEY = "features"
    # Effective width (excluding padding) of extracted features of shape (B,)
    FEATURES_WIDTH_KEY = "features_width"

    def __init__(self, vocab_size: int, input_height: int, input_channels: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_height = input_height
        self.input_channels = input_channels

    @staticmethod
    @abstractmethod
    def get_label_encoder_name() -> str:
        """
        Returns label encoding name which should be used for this head type
        """
        pass

    @staticmethod
    @abstractmethod
    def get_decoding_tensor_key() -> str:
        """
        Returns key of tensor, which should be used for decoding into string
        """
        pass

    @abstractmethod
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process some feature data to form prediction"""
        pass
