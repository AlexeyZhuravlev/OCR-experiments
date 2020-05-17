import collections
from catalyst.dl import ConfigExperiment
from .data import DatasetRegistry, PaddedDatasetWithTransforms
from .data import pre_transforms, post_transforms
from .label_encodings import FACTORY as label_encoding_factory
from .callbacks import OcrMetricsCallback, EncodeLabelsCallback, DecodeLabelsCallback
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
            datasets[dataset] = PaddedDatasetWithTransforms(raw_dataset, transforms,
                                                            image_resize_params["max_width"])

        return datasets

    def get_callbacks(self, stage: str) -> "OrderedDict[Callback]":
        """
        Registers additional callbacks for each model head:
            DecodeLabelsCallback - for transforming head predictions into strings
            OcrMetricsCallback - for metrics calculation on top of this strings
            EncodeLabelsCallback - label encoding for loss calculation (for non-inference stages)
        """
        callbacks = super().get_callbacks(stage)

        model = self.get_model(stage)

        all_label_encoders = set()
        for head_key, head in model.prediction_heads.items():
            label_encoder_name = head.get_label_encoder_name()
            all_label_encoders.add(label_encoder_name)

            prediction_key = "{}.{}".format(head_key, head.get_decoding_tensor_key())
            prediction_string_key = "{}.string".format(head_key)
            decode_callback = DecodeLabelsCallback(label_encoder_name, prediction_key,
                                                   prediction_string_key)
            decode_callback_key = "{}.decode".format(head_key)
            callbacks[decode_callback_key] = decode_callback

            ocr_metrics_callback = OcrMetricsCallback(head_key, prediction_string_key)
            ocr_metrics_callback_key = "{}.metrics".format(head_key)
            callbacks[ocr_metrics_callback_key] = ocr_metrics_callback

        if not stage.startswith("infer"):
            for label_encoder in all_label_encoders:
                encode_callback = EncodeLabelsCallback(label_encoder)
                encode_callback_key = "{}.encode".format(label_encoder)
                callbacks[encode_callback_key] = encode_callback

        return callbacks
