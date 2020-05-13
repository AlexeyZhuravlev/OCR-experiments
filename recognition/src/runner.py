from catalyst.dl import SupervisedRunner
from typing import Any, Mapping
from .data import DataItemKeys

class AdditionalDataKeys:
    """
    Additional keys for specific prediction heads parameters
    """
    # Key main model dict to store additional parameters
    HEADS_ADDITIONAL_DATA = "heads_additional"
    # Additional keys, which are added to model input whet attention_heads=True
    TEACHER_FORCING_LABELS_KEY = "teacher_forcing_labels"
    ATTENTION_NUM_STEPS_KEY = "attention_num_steps"

class OcrRunner(SupervisedRunner):
    def __init__(self, model=None, device=None, attention_heads=False,
                 max_prediction_length=25, attention_labels_key=""):
        super().__init__(model=model, device=device,
                         input_key=None, output_key=None, input_target_key=None)
        self.attention_heads = attention_heads
        self.max_prediction_length = max_prediction_length
        self.attention_labels_key = attention_labels_key

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        model_input = {
            DataItemKeys.IMAGE: batch[DataItemKeys.IMAGE],
            DataItemKeys.IMAGE_WIDTH: batch[DataItemKeys.IMAGE_WIDTH],
            AdditionalDataKeys.HEADS_ADDITIONAL_DATA: {}
        }

        # Update additional data here
        if self.attention_heads:
            attention_heads_data = self._get_attention_heads_input(batch)
            model_input[AdditionalDataKeys.HEADS_ADDITIONAL_DATA].update(attention_heads_data)

        output = self.model(model_input, **kwargs)

        return output

    def _get_attention_heads_input(self, batch):
        # Add teacher forcing labels during training and None with target length othervise
        if self.state.is_train_loader:
            labels = batch[self.attention_labels_key]
            num_steps = labels.shape[1]
            return {
                AdditionalDataKeys.TEACHER_FORCING_LABELS_KEY: labels,
                AdditionalDataKeys.ATTENTION_NUM_STEPS_KEY: num_steps
            }
        else:
            return {
                AdditionalDataKeys.TEACHER_FORCING_LABELS_KEY: None,
                AdditionalDataKeys.ATTENTION_NUM_STEPS_KEY: self.max_prediction_length
            }
