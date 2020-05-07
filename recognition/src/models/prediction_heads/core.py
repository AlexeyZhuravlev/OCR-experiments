import torch
from torch import nn
from abc import ABC, abstractmethod
from ..label_encoding import BaseLabelEncoding

class OcrPredictionHead(ABC, nn.Module):
    def __init__(self, vocab_path: str, input_height: int, input_channels: int):
        super().__init__()

        self.vocab_path = vocab_path
        self.input_height = input_height
        self.input_channels = input_channels

    @property
    @abstractmethod
    def label_encoding(self) -> BaseLabelEncoding:
        """Specific label encoding method for this head"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
