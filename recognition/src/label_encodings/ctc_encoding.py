import torch
from typing import Dict, List

from .core import BaseLabelEncoding

class CtcLabelEncoding(BaseLabelEncoding):
    """Label encoding used for models based on CTC-loss training"""

    # Token representing blank, used for CTC loss calculation
    BLANK_TOKEN = 0
    # Keys of encoded targets: labels and label lengths
    LABELS_KEY = "ctc.labels"
    LABEL_LENGTHS_KEY = "ctc.label_lengths"

    def __init__(self, vocab):
        super().__init__(vocab, aux_tokens_front=1)

        self.expected_output_channels = self.vocabulary_size + 1

    def encode_targets(self, strings: List[str]) -> Dict[str, torch.Tensor]:
        """Concatenates all targets and return their labels with lengths"""
        total_length = len("".join(strings))
        number_of_strings = len(strings)
        labels = torch.LongTensor(total_length)
        label_lengths = torch.LongTensor(number_of_strings)
        pos = 0
        for i, string in enumerate(strings):
            label_lengths[i] = len(string)
            for symbol in string:
                labels[pos] = self.char_to_label[symbol]
                pos += 1

        return {self.LABELS_KEY: labels, self.LABEL_LENGTHS_KEY: label_lengths}

    def decode_predictions(self, prediction: torch.Tensor) -> List[str]:
        """
        Implementation of CTC greedy decoding here.
        Expects (L, B, C) input tensor consisting of B sequences of length L with C log_probs
        """
        max_values = prediction.argmin(axis=-1)
        length, batch_size = max_values.shape()
        result = []
        for element in range(batch_size):
            current_str = ""
            for pos in range(length):
                current_label = max_values[pos][element]
                # Skip blanks and repeated symbols
                if current_label != self.BLANK_TOKEN and \
                   (pos == 0 or current_label != max_values[pos - 1][element]):
                    current_str += self.label_to_char[current_label]
            result.append(current_str)
        return result
