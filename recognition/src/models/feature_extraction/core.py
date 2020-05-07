import torch
from torch import nn
from torchvision.models import resnet
from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC, nn.Module):
    """Base class for all feature extractors"""
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def vertical_scale(self) -> int:
        """Vertical scale of the model: how much height changes"""
        pass

    @property
    @abstractmethod
    def horizontal_scale(self) -> int:
        """Horizontal scale of the model: how much width changes"""
        pass

    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Number of output channels for the model"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass
