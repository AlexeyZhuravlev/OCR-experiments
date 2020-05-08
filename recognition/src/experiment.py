import collections
from catalyst.dl import ConfigExperiment
from data import DatasetRegistry, DatasetWithTransforms
import albumentations as A

from typing import Dict, List

class OcrExperiment(ConfigExperiment):

    def __init__(self, config):
        super().__init__(config)

        data_registry_params = config["data_registry"]
        self.data_registry = DatasetRegistry(data_registry_params["rootdir"])

    def get_transforms(self, stage: str, dataset: str, resize_params: Dict):
        return None

    def get_datasets(self, stage: str, dataset_names: Dict, image_resize_params: Dict):
        datasets = collections.OrderedDict()

        for dataset, registry_key in dataset_names.items():
            raw_dataset = self.data_registry.get(registry_key)
            transforms = self.get_transforms(stage, dataset, image_resize_params)
            datasets[dataset] = DatasetWithTransforms(raw_dataset, transforms)

        return datasets
