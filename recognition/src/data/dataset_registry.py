"""
All datasets, used for training and testing to incapsulate all paths to dataset logic any further
"""

import os
from typing import List, Dict
import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset, Sampler
from .dataset import LmdbDataset
from .balancing import build_balanced_concatenation

class OperationTypeKeys:
    """
    Different operation types on top of registered dataset,
    which can be set in params config
    """
    # Split dataset into two by given ratio
    SPLIT = "split"
    # Merge several datasets into one
    MERGE = "merge"
    # Merge dataset and accociate sampler object which should perform
    # balanced loading from different components
    BALANCED_MERGE = "balanced_merge"

class DatasetRegistry:
    """
    Registry for all datasets
    """

    def __init__(self, params: Dict):
        self.data_root = params["rootdir"]
        self.case_sensitive = params["case_sensitive"]
        self.register = {}
        self.samplers = {}
        self.register_all_paths(params.get("path_data", {}))
        self.register_operations_data(params.get("operations", []))

    def get(self, name: str) -> Dataset:
        """
        Returns dataset by registered name
        """
        return self.register[name]

    def get_sampler(self, name: str) -> Sampler:
        """
        Returns specific sampler, assotiated with corresponding dataset
        If sampler doesn't exist, returns None
        """
        if name in self.samplers:
            return self.samplers[name]
        else:
            return None

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
        elif operation_type == OperationTypeKeys.MERGE:
            self.register_merge(**kwargs)
        elif operation_type == OperationTypeKeys.BALANCED_MERGE:
            self.register_balanced_merge(**kwargs)
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

    def register_merge(self, target: str, components: List[str]):
        """
        Registers merge of several datasets
        """
        datasets = [self.register[name] for name in components]
        merged_dataset = ConcatDataset(datasets)

        self._register_data_element(target, merged_dataset)

    def register_balanced_merge(self, target: str, components: Dict[str, float],
                                mini_epoch_len: int):
        """
        Registers balanced merge given datasets with associated weights
        (registers both dataset and corresponding sampler in class)
        """
        datasets_with_weights = []
        for name, weight in components.items():
            datasets_with_weights.append((self.register[name], weight))
        dataset, sampler = build_balanced_concatenation(datasets_with_weights, mini_epoch_len)

        self._register_data_element(target, dataset)
        self.samplers[target] = sampler

    def _get_dataset(self, relative_path):
        "Returns dataset by relative path"
        data_root = self.data_root
        full_path = os.path.join(data_root, relative_path)

        return LmdbDataset(full_path, case_sensitive=self.case_sensitive)

    def _register_data_element(self, name, dataset):
        self.register[name] = dataset

    def _register_by_path(self, name, path):
        self._register_data_element(name, self._get_dataset(path))
