# TODO: support rnn-encoder based on "squeezed"" LSTMs like in
# https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/sequence_modeling.py
# Also add this encoder support to CTC head and attention head

from torch import nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

class SqueezedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SqueezedBiLSTM, self).__init__()
        self.lstms = nn.ModuleList(
            BidirectionalLSTM(input_size, hidden_size, hidden_size)
        )
        for _ in range(num_layers - 1):
            self.lstms.append(
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
            )

    def forward(self, x):
        for layer in self.lstms:
            x = layer(x)

        return x
