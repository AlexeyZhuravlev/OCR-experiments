import torch
from typing import Dict, List

from .core import BaseLabelEncoding

class SequenceLabelEncoding(BaseLabelEncoding):
    """Label encoding used for seq2seq models with cross-entropy loss"""
    # End-of-sequence token
    EOS_TOKEN = 0
    # Padding token, added on the right for shorter sequences
    # (default ignore_index value for torch.nn.CrossEntropyLoss)
    PAD_TOKEN = -100

    # Key for encoded targets
    LABELS_KEY = "seq.labels"

    def __init__(self, vocab):
        super().__init__(vocab, aux_tokens_front=1)


    def encode_targets(self, strings: List[str]) -> Dict[str, torch.Tensor]:
        """Encodes strings as tensor of shape (batch_size, seq_length)"""
        # Max string length +1 for EOS token
        result_length = len(max(strings, key=len)) + 1
        batch_size = len(strings)
        result = torch.LongTensor(batch_size, result_length).fill_(self.PAD_TOKEN)
        for string_num, string in enumerate(strings):
            for symbol_num, symbol in enumerate(string):
                result[string_num][symbol_num] = self.char_to_label[symbol]
            result[string_num][len(string)] = self.EOS_TOKEN

        return {self.LABELS_KEY: result}

    def decode_predictions(self, prediction: torch.Tensor) -> List[str]:
        """Decodes prediction of shape (batch_size, num_classes, seq_length)"""
        prediction = prediction.argmax(axis=1)
        batch_size, length = prediction.shape
        result = []
        for element in range(batch_size):
            current_string = ""
            for position in range(length):
                label = prediction[element][position].item()
                if label == self.EOS_TOKEN:
                    break
                current_string += self.label_to_char[label]
            result.append(current_string)

        return result
    