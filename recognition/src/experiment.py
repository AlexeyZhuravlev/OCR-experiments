import collections
from catalyst.dl import ConfigExperiment
from .data import DatasetRegistry, DatasetWithTransforms
from .data import pre_transforms, post_transforms
from .label_encodings import FACTORY as label_encoding_factory
import albumentations as A
import codecs

from typing import Dict, List

class OcrExperiment(ConfigExperiment):

    def __init__(self, config):
        super().__init__(config)

        data_registry_params = config["data_registry"]
        self.data_registry = DatasetRegistry(data_registry_params)

        self._init_vocab(config)

    def _init_vocab(self, config):
        vocab_path = config["vocab"]
        vocab_file = codecs.open(vocab_path, encoding="utf-8")
        vocab = vocab_file.read()
        self.vocab = vocab
        vocab_file.close()
        # Set vocabulary for label encoding factory
        label_encoding_factory.set_vocab(self.vocab)
        # Save vocabulary size for model creation params (to obtain correct heads)
        self._config["model_params"]["vocab_size"] = len(self.vocab)

    def get_transforms(self, stage: str, dataset: str, resize_params: Dict):
        return A.Compose([pre_transforms(**resize_params), post_transforms()])

    def get_datasets(self, stage: str, dataset_names: Dict, image_resize_params: Dict):
        datasets = collections.OrderedDict()

        for dataset, registry_key in dataset_names.items():
            raw_dataset = self.data_registry.get(registry_key)
            transforms = self.get_transforms(stage, dataset, image_resize_params)
            datasets[dataset] = DatasetWithTransforms(raw_dataset, transforms)

        return datasets
