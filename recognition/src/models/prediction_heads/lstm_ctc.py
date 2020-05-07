import torch
from torch import nn
from typing import Dict, List
from ..label_encoding import BaseLabelEncoding, CtcLabelEncoding
from .core import OcrPredictionHead

class LstmCtcPredictionHead(OcrPredictionHead):
    """
    Prediction head based on LSTM along width dimension of the tensor
    """
    def __init__(self, base_params, lstm_args: Dict):
        super().__init__(**base_params)

        self.rnn = nn.LSTM(self.input_height * self.input_channels, **lstm_args)
        self._label_encoding = CtcLabelEncoding(self.vocab_path)

        num_directions = 2 if self.rnn.bidirectional else 1
        self.linear = nn.Linear(self.rnn.hidden_size * num_directions,\
                                self._label_encoding.expected_output_channels)

    @property
    def label_encoding(self) -> BaseLabelEncoding:
        """Specific label encoding method for this head"""
        return self._label_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        assert height == self.input_height
        assert channels == self.input_channels

        x = x.view(batch, channels * height, width)
        x = x.permute(2, 0, 1)

        x, _ = self.rnn(x)
