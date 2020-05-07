from typing import Dict
from .core import OcrPredictionHead

class PredictionHeadsFactory:
    def __init__(self):
        self.prediction_heads = {}
        self.base_params = None

    def set_base_params(self, base_params: Dict):
        self.base_params = base_params

    def register(self, name: str, head):
        self.prediction_heads[name] = head

    def get(self, name: str, specific_args: Dict) -> OcrPredictionHead:
        """Returns instantiated head given it's parameters. base_params should be set before it"""
        if not self.base_params:
            raise RuntimeError("Base params for PredictionHeadsFacory not initialized")
        head = self.prediction_heads.get(name)
        if not head:
            raise ValueError(name)
        return head(self.base_params, **specific_args)
