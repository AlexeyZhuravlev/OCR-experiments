from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import OcrPredictionHead
from src.runner import AdditionalDataKeys

class AttentionCell(nn.Module):
    """
    Attention cell implementation: evaluating hidden size at each time step
    based on
    """
    def __init__(self, input_size, hidden_size, token_input_size):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + token_input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_h, prev_token):
        """
        input:
            prev_hidden - internal backend recurrent cell previous hidden state
            batch_h - contextual features to take attention on (batch_size x seq_len x input_size)
            prev_token - encoded previous token of shape (batch_size x element_input_size)
        output:
            new hidded state of backend recurrent cell
        """
        # [batch_size x seq_len x input_size] -> [batch_size x seq_len x hidden_size]
        batch_h_proj = self.i2h(batch_h)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        # batch_size x seq_len x 1
        e = self.score(torch.tanh(batch_h_proj + prev_hidden_proj))

        alpha = F.softmax(e, dim=1)
        # context vector: batch_size x input_size
        context = torch.bmm(alpha.permute(0, 2, 1), batch_h).squeeze(1)
        # batch_size x (num_channel + num_embedding)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden

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

    def forward(self, context_features, teacher_forcing_labels, num_steps):
        """
        input:
            context_features hidden state of encoder (batch_size x seq_len x input_size)
            teacher_forcing_labels : [num_steps x batch_size]. Can be None, i.e. on inference
        output: log probability distribution at each step [num_steps x batch_size x num_classes]
        """
        batch_size = context_features.size(0)
        device = context_features.device()

        # LSTM initial hidden state and cell
        prev_hidden = (torch.zeros(batch_size, self.hidden_size).to(device),
                       torch.zeros(batch_size, self.hidden_size).to(device))
        # Start from SOS token
        prev_token = torch.LongTensor(batch_size).fill(self.sos_token).to(device)

        result = torch.zeros(num_steps, batch_size, self.output_size)

        for i in range(num_steps):
            char_inputs = self.embedding(prev_token)
            current_hidden = self.attention_cell(prev_hidden, context_features, char_inputs)
            generator_input = current_hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            # Current probs of shape (batch_size x num_classes)
            current_logits = self.generator(generator_input)
            # Store result
            result[i] = F.log_softmax(current_logits, dim=-1)

            prev_hidden = current_hidden
            if teacher_forcing_labels:
                prev_token = teacher_forcing_labels[i]
            else:
                prev_token = current_logits.argmax(axis=-1)

        return result

class RnnAttentionHead(OcrPredictionHead):
    """
    Attention head based on LSTM encoder and LSTM decoder
    """
    # Keys in result dictionary
    LOG_PROBS_KEY = "logprobs"

    def __init__(self, base_params, encoder_lstm_args, decoder_hidden_size, embedding_size):
        super().__init__(**base_params)

        encoder_lstm_args.extend(batch_first=True)
        self.encoder = nn.LSTM(self.input_height * self.input_channels, **encoder_lstm_args)
        self.decoder = RnnAttentionDecoder(self.encoder.hidden_size, decoder_hidden_size,
                                           self.vocab_size, embedding_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = data[self.FEATURES_KEY]
        teacher_forcing_labels = data[AdditionalDataKeys.TEACHER_FORCING_LABELS_KEY]
        num_steps = data[AdditionalDataKeys.ATTENTION_NUM_STEPS_KEY]

        batch, channels, height, width = features.shape
        assert height == self.input_height
        assert channels == self.input_channels

        features = features.view(batch, channels * height, width)
        features = features.permute(0, 2, 1)
        features_encoded = self.encoder(features)
        probabilities = self.decoder(features_encoded, teacher_forcing_labels, num_steps)

        return {
            self.LOG_PROBS_KEY: probabilities
        }
