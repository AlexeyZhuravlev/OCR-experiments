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
    # Additional key, which are added to model input when it has attention head
    TEACHER_FORCING_LABELS_KEY = "teacher_forcing_labels"

class OcrRunner(SupervisedRunner):
    """
    Main runner class for OCR experiments.
    Handles additional logic about which data is passed to model
    """

    def __init__(self, model=None, device=None):
        super().__init__(model=model, device=device,
                         input_key=None, output_key=None, input_target_key=None)

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        model_input = {
            DataItemKeys.IMAGE: batch[DataItemKeys.IMAGE],
            DataItemKeys.IMAGE_WIDTH: batch[DataItemKeys.IMAGE_WIDTH],
            AdditionalDataKeys.HEADS_ADDITIONAL_DATA: {}
        }

        # Add additional data for some classification heads
        if self.model.has_attention_head:
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
            return {
                AdditionalDataKeys.TEACHER_FORCING_LABELS_KEY: labels.to(self.device)
            }
        else:
            return {
                AdditionalDataKeys.TEACHER_FORCING_LABELS_KEY: None
            }
