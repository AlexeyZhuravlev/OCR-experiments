import codecs
import torch
from typing import Dict, List
from abc import ABC, abstractmethod

class BaseLabelEncoding(ABC):
    """Class responsible for encoding strings into integers and decoding them back"""

    def __init__(self, vocab_path, aux_tokens_front=0):
        """Gives aux_tokens_count elements in the front for encoding"""
        vocab_file = codecs.open(vocab_path, encoding="utf-8")
        separate_characters = vocab_file.read()
        vocab_file.close()

        self.char_to_label = dict()
        self.label_to_char = dict()
        self.vocabulary_size = len(separate_characters)
        self.aux_tokens_front = aux_tokens_front

        for i, symbol in enumerate(separate_characters):
            self.char_to_label[symbol] = i + aux_tokens_front
            self.label_to_char[i + aux_tokens_front] = symbol

    @abstractmethod
    def encode_targets(self, strings: List[str]) -> Dict[str, torch.Tensor]:
        """Encodes list of groundtruth string as specific criterion targets for training"""
        pass

    @abstractmethod
    def decode_predictions(self, prediction: torch.Tensor) -> List[str]:
        """Decodes model prediction as list of strings"""
        pass

class CtcLabelEncoding(BaseLabelEncoding):
    """Label encoding used for models based on CTC-loss training"""

    # Token representing blank, used for CTC loss calculation
    BLANK_TOKEN = 0
    # As padding token we just reuse blank as loss functions ignores padding anyway
    PAD_TOKEN = 0
    # Keys of encoded targets vocabulary: labels and label lengths
    LABELS_KEY = "labels"
    LABEL_LENGTHS_KEY = "label_lengths"

    def __init__(self, vocab_path):
        super().__init__(vocab_path, aux_tokens_front=1)

        self.expected_output_channels = self.vocabulary_size + 1

    def encode_targets(self, strings: List[str]) -> Dict[str, torch.Tensor]:
        """Encodes strings as tensor of padded long labels with additional lengths tensor"""
        max_length = len(max(strings, key=len))
        number_of_strings = len(strings)
        labels = torch.LongTensor(number_of_strings, max_length).fill_(self.PAD_TOKEN)
        label_lengths = torch.LongTensor(number_of_strings)

        for i, string in enumerate(strings):
            label_lengths[i] = len(string)
            for pos, symbol in enumerate(string):
                labels[i][pos] = self.char_to_label[symbol]

        return {self.LABELS_KEY: labels, self.LABEL_LENGTHS_KEY: label_lengths}

    def decode_predictions(self, prediction: torch.Tensor) -> List[str]:
        """
        Implementation of CTC greedy decoding here.
        Expects (L, B, C) input tensor consisting of B sequences of length L with C logits
        """
        max_values = prediction.argmax(axis=-1)
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
