from .core import BaseLabelEncoding

class LabelEncodingFactory:
    def __init__(self):
        self.label_encoders = {}
        self.vocabulary = None

    def set_vocab(self, vocabulary):
        self.vocabulary = vocabulary

    def register(self, name: str, encoder):
        self.label_encoders[name] = encoder

    def get(self, name: str) -> BaseLabelEncoding:
        encoder = self.label_encoders.get(name)
        if not encoder:
            raise ValueError(name)
        return encoder(self.vocabulary)
