import torch
from typing import Dict, List
from abc import ABC, abstractmethod

class BaseLabelEncoding(ABC):
    """Class responsible for encoding strings into integers and decoding them back"""

    def __init__(self, separate_characters, aux_tokens_front=0):
        """Gives aux_tokens_count elements in the front for encoding"""

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
