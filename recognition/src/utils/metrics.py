"""
Metric functions used for model evaluaion
"""

from typing import List
import editdistance
import re

class AlphanumPreprocessing:
    """
    String preprocessing strategy which removes non-alphanumerics from string
    """
    def __init__(self):
        self.pattern = re.compile(r"[^a-zA-Z0-9]")

    def __call__(self, string: str):
        return self.pattern.sub("", string)


class GroupOcrMetrics:
    """
    Group of several OCR metrics, which are evaluated depending on string preprocessing rules:
    i.e. compare only alphanumeric parts of the strings
    """
    # Complete string comparison
    FULL_PREFIX = "full."
    # Comparison only on alphanumeric level (all other symbols are deleted)
    ALPHANUM_PREFIX = "alphanum."

    def __init__(self):
        self.metrics = []
        self._add_group(self.FULL_PREFIX, None)
        self._add_group(self.ALPHANUM_PREFIX, AlphanumPreprocessing())

    def _add_group(self, prefix, preprocessing):
        self.metrics.append(OcrMetrics(prefix=prefix, preprocessing_fn=preprocessing))

    def reset_counters(self):
        for metrics in self.metrics:
            metrics.reset_counters()

    def add_batch(self, prediction_strings, groundtruth_strings):
        for metrics in self.metrics:
            metrics.add_batch(prediction_strings, groundtruth_strings)

    def get_metrics(self):
        result = {}
        for metrics in self.metrics:
            result.update(metrics.get_metrics())

        return result

class OcrMetrics:
    """
    Main class for predictions registration and metrics accumulation
    """
    CHAR_ACCURACY_KEY = "char_acc"
    FRAGMENT_ACCURACY_KEY = "fragment_acc"

    def __init__(self, prefix="", preprocessing_fn=None):
        self.prefix = prefix
        self.preprocessing_fn = preprocessing_fn
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

        if self.preprocessing_fn is not None:
            prediction = self.preprocessing_fn(prediction)
            groundtruth = self.preprocessing_fn(groundtruth)

        if prediction == groundtruth:
            self.correct_fragments += 1
        distance = editdistance.eval(prediction, groundtruth)
        max_len = max(len(prediction), len(groundtruth))
        if max_len == 0:
            # Both predictions are empty: total match
            norm_distance = 1.
        else:
            norm_distance = distance / max(len(prediction), len(groundtruth))
        self.total_ned += norm_distance

    def get_metrics(self):
        """
        Return accumulated metric values as a dictionary
        """
        fragment_accuracy = self.correct_fragments / self.total_fragments
        character_accuracy = 1. - self.total_ned / self.total_fragments

        result = {
            self.prefix + self.CHAR_ACCURACY_KEY: character_accuracy,
            self.prefix + self.FRAGMENT_ACCURACY_KEY: fragment_accuracy
        }

        return result
