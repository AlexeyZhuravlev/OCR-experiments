from catalyst.dl import SupervisedRunner
from typing import Any, Mapping
from .data import DataItemKeys
from .label_encodings.seq_encoding import SequenceLabelEncoding

class AdditionalDataKeys:
    """
    Additional keys for specific prediction heads parameters
    """
    # Key main model dict to store additional parameters
    HEADS_ADDITIONAL_DATA = "heads_additional"
    # Keys, which are used inside this dictionary
    # Additional keys, which are added to model input whet attention_heads=True
    TEACHER_FORCING_LABELS_KEY = "teacher_forcing_labels"
    ATTENTION_NUM_STEPS_KEY = "attention_num_steps"

class OcrRunner(SupervisedRunner):
    """
    Main runner class for OCR experiments.
    Handles additional logic about which data is passed to model
    """

    def __init__(self, model=None, device=None, attention_heads=False,
                 max_prediction_length=25):
        super().__init__(model=model, device=device,
                         input_key=None, output_key=None, input_target_key=None)
        self.attention_heads = attention_heads
        self.max_prediction_length = max_prediction_length

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        model_input = {
            DataItemKeys.IMAGE: batch[DataItemKeys.IMAGE],
            DataItemKeys.IMAGE_WIDTH: batch[DataItemKeys.IMAGE_WIDTH],
            AdditionalDataKeys.HEADS_ADDITIONAL_DATA: {}
        }

        # Add additional data for some classification heads
        if self.attention_heads:
            attention_heads_data = self._get_attention_heads_input(batch)
            model_input[AdditionalDataKeys.HEADS_ADDITIONAL_DATA].update(attention_heads_data)

        output = self.model(model_input, **kwargs)

        return output

    def _get_attention_heads_input(self, batch):
        # Add teacher forcing labels during training
        # None with specified in paratemers target length is passed during val/inference
        if self.state.is_train_loader:
            # Sequence labels encoded as (batch_size, sequence_length) - valid for loss calculation
            labels = batch[SequenceLabelEncoding.LABELS_KEY]
            # transpose to (sequence_length, batch_size) - heads expect this shape
            labels = labels.transpose(1, 0)
            num_steps = labels.shape[0]
            return {
                AdditionalDataKeys.TEACHER_FORCING_LABELS_KEY: labels.to(self.device),
                AdditionalDataKeys.ATTENTION_NUM_STEPS_KEY: num_steps
            }
        else:
            return {
                AdditionalDataKeys.TEACHER_FORCING_LABELS_KEY: None,
                AdditionalDataKeys.ATTENTION_NUM_STEPS_KEY: self.max_prediction_length
            }
