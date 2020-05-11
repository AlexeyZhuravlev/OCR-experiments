"""
All datasets, used for training and testing to incapsulate all paths to dataset logic any further
"""

import os
from torch.utils.data import ConcatDataset, Dataset
from .dataset import LmdbDataset
from typing import Dict

class DatasetRegistry:
    """
    Registry for all dataset
    """

    def __init__(self, params: Dict):
        self.data_root = params["rootdir"]
        self.register = {}
        self._register_all_data()

    def get(self, name: str) -> Dataset:
        """
        Returns dataset by registered name
        """
        return self.register[name]

    def _get_dataset(self, relative_path):
        "Returns dataset by relative path"
        data_root = self.data_root
        full_path = os.path.join(data_root, relative_path)

        return LmdbDataset(full_path)

    def _register_data_element(self, name, dataset):
        self.register[name] = dataset

    def _register_by_path(self, name, path):
        self._register_data_element(name, self._get_dataset(path))

    def _register_all_data(self):
        self._register_by_path("MJSYNTH", "Synthetic/MJ")
        synthtext_usual = self._get_dataset("Synthetic/ST_AN")
        synthtext_punct = self._get_dataset("Synthetic/ST_spe")
        self._register_data_element("SYNTHTEXT", ConcatDataset([synthtext_usual, synthtext_punct]))
        self._register_by_path("SYNTHADD", "Synthetic/SynthAdd")

        self._register_by_path("COCO_TRAIN", "Real/Train/COCO_TRAIN")
        self._register_by_path("COCO_VAL", "Real/Train/COCO_VAL")
        self._register_by_path("IC03_TRAIN", "Real/Train/IC03")
        self._register_by_path("IC13_TRAIN", "Real/Train/IC13")
        self._register_by_path("IC15_TRAIN", "Real/Train/IC15")
        self._register_by_path("IIIT5K_TRAIN", "Real/Train/IIIT5k")
        self._register_by_path("RRC_ART", "Real/Train/RRC_ArT")
        self._register_by_path("SVT_TRAIN", "Real/Train/SVT")

        self._register_by_path("IC03_TEST", "Real/Test/IC03")
        self._register_by_path("IC13_TEST", "Real/Test/IC13")
        self._register_by_path("IC15_TEST", "Real/Test/IC15")
        self._register_by_path("IIIT5K_TEST", "Real/Test/IIIT5K")
        self._register_by_path("SVT_TEST", "Real/Test/SVT")
        self._register_by_path("SVTP", "Real/Test/SVTP")
        self._register_by_path("CUTE80", "Real/Test/CUTE80")

        self._register_by_path("CAR_TRAIN", "Sample/Train")
        self._register_by_path("CAR_TEST", "Sample/Test")
