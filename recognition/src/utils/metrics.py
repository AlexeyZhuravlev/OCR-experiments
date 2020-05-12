"""
Metric functions used for model evaluaion
"""

from typing import List
import editdistance

class OcrMetrics:
    """
    Main class for predictions registration and metrics accumulation
    """
    CHAR_ACCURACY_KEY = "char_acc"
    FRAGMENT_ACCURACY_KEY = "fragment_acc"

    def __init__(self):
        self.reset_counters()

    def reset_counters(self):
        """
        Resets counters and prepare for new metric calculations
        """
        self.total_fragments = 0
        self.correct_fragments = 0
        self.total_ned = 0

    def add_batch(self, prediction_strings: List[str], groundtruth_strings: List[str]):
        """
        Register batch of samples
        """
        assert len(prediction_strings) == len(groundtruth_strings)

        for prediction, groundtruth in zip(prediction_strings, groundtruth_strings):
            self.add_sample(prediction, groundtruth)

    def add_sample(self, prediction: str, groundtruth: str):
        """
        Register single sample
        """
        self.total_fragments += 1

        if prediction == groundtruth:
            self.correct_fragments += 1
        distance = editdistance.eval(prediction, groundtruth)
        norm_distance = distance / max(len(prediction), len(groundtruth))
        self.total_ned += norm_distance

    def get_metrics(self):
        """
        Return accumulated metric values as a dictionary
        """
        fragment_accuracy = self.correct_fragments / self.total_fragments
        character_accuracy = 1. - self.total_ned / self.total_fragments

        result = {
            self.CHAR_ACCURACY_KEY: character_accuracy,
            self.FRAGMENT_ACCURACY_KEY: fragment_accuracy
        }

        return result
