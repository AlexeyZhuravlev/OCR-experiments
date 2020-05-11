from catalyst.dl import SupervisedRunner
from typing import Any, Mapping
from .data import DataItemKeys

class OcrRunner(SupervisedRunner):
    def __init__(self, model=None, device=None):
        super().__init__(model=model, device=device,
                         input_key=None, output_key=None, input_target_key=None)

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        model_input = {
            DataItemKeys.IMAGE: batch[DataItemKeys.IMAGE],
            DataItemKeys.IMAGE_WIDTH: batch[DataItemKeys.IMAGE_WIDTH]
        }
        output = self.model(model_input, **kwargs)

        return output
