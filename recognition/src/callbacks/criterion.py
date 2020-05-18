from torch import nn

from catalyst.core.callbacks import CriterionCallback
from catalyst.core import State
from src.models.prediction_heads import HeadOutputKeys
from src.label_encodings import CtcLabelEncoding, SequenceLabelEncoding

class CtcCriterionCallback(CriterionCallback):
    """
    Callback for CTC Loss calculation on specified head
    """
    def __init__(self, head_key, multiplier=1.0, loss_suffix="loss"):
        # Binding between state keys and torch.nn.CTCLoss parameters
        # Groundtruth + groundtruth length
        targets = CtcLabelEncoding.LABELS_KEY
        target_lengths = CtcLabelEncoding.LABEL_LENGTHS_KEY
        # Prediction + prediction length
        log_probs = "{}.{}".format(head_key, HeadOutputKeys.LOG_PROBS)
        input_lengths = "{}.{}".format(head_key, HeadOutputKeys.LOG_PROBS_LEN)
        input_key = {
            targets: "targets",
            target_lengths: "target_lengths"
        }
        output_key = {
            log_probs: "log_probs",
            input_lengths: "input_lengths"
        }
        loss_key = "{}.{}".format(head_key, loss_suffix)

        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=loss_key,
            multiplier=multiplier
        )

        self._criterion = nn.CTCLoss(blank=CtcLabelEncoding.BLANK_TOKEN, zero_infinity=True)

    def on_stage_start(self, state: State):
        # Override basic behaviour of CriterionCallback
        pass

class CrossEntropyCriterionCallback(CriterionCallback):
    """
    Callback for crossentropy loss calculation on specified head
    """
    def __init__(self, head_key, multiplier=1.0, loss_suffix="loss"):
        # Binding between state keys and ctc loss parameters
        logits = "{}.{}".format(head_key, HeadOutputKeys.LOGITS)
        targets = SequenceLabelEncoding.LABELS_KEY
        loss_key = "{}.{}".format(head_key, loss_suffix)

        super().__init__(
            input_key=targets,
            output_key=logits,
            prefix=loss_key,
            multiplier=multiplier
        )

        self._criterion = nn.CrossEntropyLoss(ignore_index=SequenceLabelEncoding.PAD_TOKEN)

    def on_stage_start(self, state: State):
        # Override basic behaviour of CriterionCallback
        pass
