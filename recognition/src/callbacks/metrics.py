from catalyst.core.callbacks.metrics import MetricCallback
from src.utils.metrics import fragment_accuracy, character_accuracy

class FragmentAccuracyCallback(MetricCallback):
    """
    Amount of correctly-classified fragments
    """
    def __init__(self, prefix: str, input_key: str, output_key: str):
        super().__init__(
            prefix=prefix,
            metric_fn=fragment_accuracy,
            input_key=input_key,
            output_key=output_key,
            multiplier=1.0)

class CharAccuracyCallback(MetricCallback):
    """
    Average 1 - N.E.D
    """
    def __init__(self, prefix: str, input_key: str, output_key: str):
        super().__init__(
            prefix=prefix,
            metric_fn=character_accuracy,
            input_key=input_key,
            output_key=output_key
        )
