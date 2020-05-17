import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Fully-check and debug this cell!!!
# Refactor: move-out attention scoring mechanism (concat from https://arxiv.org/pdf/1508.04025.pdf)
# Advanced TODO: GRU and other internal recurrents support
class AttentionCell(nn.Module):
    """
    Attention recurrent cell implementation: evaluating hidden size at each time step
    Implements Bahdanau additive attention mechanism (https://arxiv.org/pdf/1409.0473.pdf)
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
        concat_context = torch.cat([context, prev_token], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden
