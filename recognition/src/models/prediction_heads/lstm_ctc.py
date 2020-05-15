import torch
from torch import nn
from typing import Dict, List
from .core import OcrPredictionHead
import torch.nn.functional as F

class LstmCtcPredictionHead(OcrPredictionHead):
    """
    Prediction head based on LSTM along width dimension of the features tensor with FC prediction
    Returns tensor of log_probs of shape (L, B, C) and their effective lengths of shape (B,),
    which can be used for ctc_loss calculation.
    """
    # Keys in result dictionary
    LOG_PROBS_KEY = "logprobs"
    LOG_PROBS_LEN_KEY = "logprobs_len"

    def __init__(self, base_params, lstm_args: Dict, grad_to_features=True):
        super().__init__(**base_params)

        self.rnn = nn.LSTM(self.input_height * self.input_channels, **lstm_args)

        num_directions = 2 if self.rnn.bidirectional else 1
        self.linear = nn.Linear(self.rnn.hidden_size * num_directions,\
                                self.vocab_size + 1)
        self.grad_to_features = grad_to_features

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = data[self.FEATURES_KEY].detach()
        features_width = data[self.FEATURES_WIDTH_KEY]

        if not self.grad_to_features:
            features = features.detach()

        batch, channels, height, width = features.shape
        assert height == self.input_height
        assert channels == self.input_channels

        features = features.view(batch, channels * height, width)
        features = features.permute(2, 0, 1)

        features, _ = self.rnn(features)
        logits = self.linear(features)
        log_probs = F.log_softmax(logits, dim=-1)

        result = {
            self.LOG_PROBS_KEY: log_probs,
            self.LOG_PROBS_LEN_KEY: features_width
        }

        return result
