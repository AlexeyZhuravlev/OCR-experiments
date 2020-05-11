from catalyst.core import Callback, CallbackOrder
from ..label_encodings import FACTORY as label_encodings_factory
from ..data import DataItemKeys

class EncodeLabelsCallback(Callback):
    """
    Takes specific label encoder by name and encodes groundtruth strings,
    adding this information to state.input
    """
    def __init__(self, encoder_name):
        super().__init__(order=CallbackOrder.Metric)

        self.encoder = label_encodings_factory.get(encoder_name)

    def on_batch_start(self, state):
        raw_strings = state.input[DataItemKeys.STRING]

        encoded_result = self.encoder.encode_targets(raw_strings)
        state.input.update(encoded_result)
