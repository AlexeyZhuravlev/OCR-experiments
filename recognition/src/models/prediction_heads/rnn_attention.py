from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import OcrPredictionHead, HeadOutputKeys
from src.runner import AdditionalDataKeys
from src.models.modules import AttentionCell
from enum import Enum

class RnnAttentionDecoder(nn.Module):
    """
    Sequence prediction module, based on contextual features and recurrent attention mechanism
    """
    def __init__(self, input_size, hidden_size, vocab_size, embedding_size):
        super().__init__()

        # Additional tokens for seq prediction. Assume that label_encoder use range 1..vocab_size
        self.eos_token = 0
        self.sos_token = vocab_size + 1
        # All elements + SOS + EOS for embeddings
        self.embedding = nn.Embedding(vocab_size + 2, embedding_size)
        self.attention_cell = AttentionCell(input_size, hidden_size, embedding_size)
        self.hidden_size = hidden_size
        # Generator predicts only vocab elements and EOS (not SOS!)
        self.output_size = vocab_size + 1
        self.generator = nn.Linear(hidden_size, self.output_size)

    def forward(self, context_features, hidden_init, teacher_forcing_labels, num_steps):
        """
        input:
            context_features hidden state of encoder (batch_size x seq_len x input_size)
            hidden_init: tuple of 2 elements of shape (batch_size x hidden_size) - lstm state
            teacher_forcing_labels : [num_steps x batch_size]. Can be None, i.e. on inference
        output: logits at each step [num_steps x batch_size x num_classes]
        """
        batch_size = context_features.size(0)
        device = context_features.device

        # LSTM initial hidden state and cell
        prev_hidden = hidden_init
        # Start from SOS token
        prev_token = torch.LongTensor(batch_size).fill_(self.sos_token).to(device)

        result = torch.zeros(num_steps, batch_size, self.output_size).to(device)

        for i in range(num_steps):
            char_inputs = self.embedding(prev_token)
            current_hidden = self.attention_cell(prev_hidden, context_features, char_inputs)
            generator_input = current_hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            # Current probs of shape (batch_size x num_classes)
            current_logits = self.generator(generator_input)
            # Store result
            result[i] = current_logits

            prev_hidden = current_hidden
            if teacher_forcing_labels is None:
                prev_token = current_logits.argmax(axis=-1)
                # Stop decoding with EOS symbol predicted when BS=1 (inference)
                if batch_size == 1 and prev_token.item() == self.eos_token:
                    break
            else:
                prev_token = teacher_forcing_labels[i]

        return result

class EncoderMode(Enum):
    """
    Different modes, in which encoder  works
    """
    # Default stategy: rnn-encoded results are passed to attention decoder
    # as in RARE and most others approaches (https://arxiv.org/pdf/1603.03915.pdf)
    PASS_TO_DECODER = 0
    # Encoder is used to initialize initial decoder state, similar to classic seq2seq,
    # Like proposed in https://arxiv.org/pdf/1811.00751.pdf
    INIT_DECODER = 1 # In this mode input features are used for decoder input
    # Encoder is not used (thus is not created)
    NONE = 2 # In this mode input features are used for decoder input

class RnnAttentionHead(OcrPredictionHead):
    """
    Attention head based on optional LSTM encoder and LSTM decoder
    Returns logprobs of shape (batch_size, num_classes, length),
    appropiate for nn.CrossEntropyLoss calculation
    """

    # String constants for configs params initialization
    ENCODER_MODES_DICT = {
        'pass': EncoderMode.PASS_TO_DECODER,
        'init': EncoderMode.INIT_DECODER,
        'none': EncoderMode.NONE
    }

    def __init__(self, base_params: Dict, encoder_params: Dict, decoder_params: Dict):
        super().__init__(**base_params)

        decoder_input_size = self._init_encoder(**encoder_params)
        decoder_params.update(input_size=decoder_input_size)
        self._init_decoder(**decoder_params)

    # Initializes decoder using given mode
    def _init_encoder(self, mode: str, args: Dict):
        encoder_mode = self.ENCODER_MODES_DICT[mode]
        self.encoder_mode = encoder_mode
        features_input_size = self.input_height * self.input_channels

        if encoder_mode == EncoderMode.NONE:
            self.encoder = None
        else:
            args.update(batch_first=True)
            self.encoder = nn.LSTM(self.input_height * self.input_channels, **args)
            self.encoder.num_directions = 2 if self.encoder.bidirectional else 1

        if encoder_mode == EncoderMode.PASS_TO_DECODER:
            encoder_output_size = self._get_encoder_output_size()
            return encoder_output_size

        return features_input_size

    def _init_decoder(self, input_size, hidden_size, embedding_size):
        self.decoder = RnnAttentionDecoder(input_size, hidden_size, self.vocab_size, embedding_size)

        if self.encoder_mode == EncoderMode.INIT_DECODER:
            self.encoder_decoder_hidden_linear = nn.Linear(self._get_encoder_output_size(),
                                                           self.decoder.hidden_size)
            self.encoder_decoder_cell_linear = nn.Linear(self._get_encoder_output_size(),
                                                          self.decoder.hidden_size)

    def _get_encoder_output_size(self):
        return self.encoder.num_directions * self.encoder.hidden_size

    @staticmethod
    def get_label_encoder_name() -> str:
        return "seq"

    @staticmethod
    def get_decoding_tensor_key() -> str:
        return HeadOutputKeys.LOGITS

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = data[self.FEATURES_KEY]
        teacher_forcing_labels = data[AdditionalDataKeys.TEACHER_FORCING_LABELS_KEY]

        if teacher_forcing_labels is None:
            num_steps = data[AdditionalDataKeys.TARGET_LENGTH_KEY]
        else:
            num_steps = teacher_forcing_labels.shape[0]

        batch, channels, height, width = features.shape
        assert height == self.input_height
        assert channels == self.input_channels

        features = features.view(batch, channels * height, width)
        features = features.permute(0, 2, 1)

        decoder_input, decoder_state = self._get_decoder_input(features)

        logits = self.decoder(decoder_input, decoder_state, teacher_forcing_labels, num_steps)
        # Repack (L, B, C) -> (B, C, L)
        logits = logits.permute(1, 2, 0)

        return {
            HeadOutputKeys.LOGITS: logits
        }

    def _get_decoder_input(self, features):
        if self.encoder is not None:
            encoder_features, encoder_state = self.encoder(features)

        if self.encoder_mode == EncoderMode.INIT_DECODER:
            state = self._get_decoder_state(encoder_state)
            return features, state

        zero_state = self._get_decoder_zero_state(features.size(0), features.device)
        if self.encoder_mode == EncoderMode.PASS_TO_DECODER:
            return encoder_features, zero_state
        else:
            return features, zero_state

    def _get_decoder_state(self, encoder_state):
        encoder_hidden, encoder_cell = encoder_state
        encoder_hidden = self._pack_encoder_rnn_state(encoder_hidden)
        encoder_cell = self._pack_encoder_rnn_state(encoder_cell)
        return (
            self.encoder_decoder_hidden_linear(encoder_hidden),
            self.encoder_decoder_cell_linear(encoder_cell)
        )

    def _pack_encoder_rnn_state(self, state):
        """
        From rnn state of shape (num_layers * num_directions, batch, hidden_size)
        returns tensor of shape (batch, hidden_size * num_directions) for last layer state
        """
        batch_size = state.size(1)
        num_directions = self.encoder.num_directions
        hidden_size = self.encoder.hidden_size
        state = state.view(self.encoder.num_layers, num_directions, batch_size, hidden_size)
        # Take last element and reshape to (batch_size, num_directions, hidden_size)
        state = state[-1].permute(1, 0, 2)
        state = state.reshape(batch_size, num_directions * hidden_size)

        return state

    def _get_decoder_zero_state(self, batch_size, device):
        hidden_size = self.decoder.hidden_size
        return (
            torch.zeros(batch_size, hidden_size).to(device),
            torch.zeros(batch_size, hidden_size).to(device)
        )
