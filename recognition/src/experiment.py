import collections
from catalyst.dl import ConfigExperiment
from data.dataset_registry import DatasetRegistry
import albumentations as A

class OcrExperiment(ConfigExperiment):

    def __init__(self, config):
        super().__init__(config)

        data_registry_params = config["data_registry"]
        self.data_registry = DatasetRegistry(data_registry_params["rootdir"])

    @staticmethod
    def get_transforms():
        return A.Normalize()

    def get_datasets(self, stage: str, **kwargs):
        datasets = collections.OrderedDict()

        data_params = self.stages_config[stage]["data_params"]
        train_dataset_key = data_params["train_key"]
        valid_dataset_key = data_params["valid_key"]
        datasets["train"] = self.data_registry.get(train_dataset_key)
        datasets["valid"] = self.data_registry.get(valid_dataset_key)

        return datasets
