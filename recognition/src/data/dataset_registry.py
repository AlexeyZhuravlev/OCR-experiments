"""
All datasets, used for training and testing to incapsulate all paths to dataset logic any further
"""

import os
from typing import List, Dict
import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from .dataset import LmdbDataset

# TODO: add more operation types like merge etc.
class OperationTypeKeys:
    """Data operation types to avoid misprints"""
    SPLIT = "split"

class DatasetRegistry:
    """
    Registry for all datasets
    """

    def __init__(self, params: Dict):
        self.data_root = params["rootdir"]
        self.register = {}
        self.register_all_paths(params.get("path_data", {}))
        self.register_operations_data(params.get("operations", []))

    def get(self, name: str) -> Dataset:
        """
        Returns dataset by registered name
        """
        return self.register[name]

    def register_all_paths(self, params: Dict):
        for key, value in params.items():
            self.register_by_path(key, value)

    def register_operations_data(self, params: List[Dict]):
        for operation_params in params:
            self.register_operation_data(**operation_params)

    def register_operation_data(self, **kwargs):
        operation_type = kwargs.pop("type")
        if operation_type == OperationTypeKeys.SPLIT:
            self.register_split(**kwargs)
        else:
            raise ValueError(operation_type)


    def register_by_path(self, name, path):
        """Registers dataset with key {name} by given path"""
        self._register_data_element(name, self._get_dataset(path))

    def register_split(self, main: str, first: str, second: str, first_size=0.5, shuffle=False):
        """
        Registers split for data element main of ratio first_size
        and saves result to first and second keys respectively.
        Shuffles indices before if needed
        """
        main_dataset = self.register[main]

        main_dataset_size = len(main_dataset)
        split_index = round(main_dataset_size * first_size)

        if shuffle:
            indices = np.random.permutation(main_dataset_size)
        else:
            indices = np.arange(main_dataset_size)

        first_dataset = Subset(main_dataset, indices[:split_index])
        second_dataset = Subset(main_dataset, indices[split_index:])

        self._register_data_element(first, first_dataset)
        self._register_data_element(second, second_dataset)


    def _get_dataset(self, relative_path):
        "Returns dataset by relative path"
        data_root = self.data_root
        full_path = os.path.join(data_root, relative_path)

        return LmdbDataset(full_path)

    def _register_data_element(self, name, dataset):
        self.register[name] = dataset

    def _register_by_path(self, name, path):
        self._register_data_element(name, self._get_dataset(path))

    # TODO: Move all these data elements to config
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
