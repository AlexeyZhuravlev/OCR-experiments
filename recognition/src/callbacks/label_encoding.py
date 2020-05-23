from catalyst.core import Callback, CallbackOrder
from src.label_encodings import FACTORY as label_encodings_factory
from src.data import DataItemKeys

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
        for key, value in encoded_result.items():
            state.input[key] = value.to(state.device)

class DecodeLabelsCallback(Callback):
    """
    Takes tensor by specific input_key in state.output and decodes it
    into list of string using specific strategy
    Stores result in the state.output by output_key
    """
    def __init__(self, encoder_name, input_key, output_key):
        super().__init__(order=CallbackOrder.Metric)

        self.encoder = label_encodings_factory.get(encoder_name)
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state):
        # Don't decode labels during training for training speedup
        if state.is_train_loader:
            return

        target_tensor = state.output[self.input_key]
        strings = self.encoder.decode_predictions(target_tensor)
        state.output[self.output_key] = strings
