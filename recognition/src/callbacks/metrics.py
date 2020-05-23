from catalyst.core.callbacks.metrics import MetricCallback
from src.utils.metrics import OcrMetrics
from src.data import DataItemKeys

class OcrMetricsCallback(MetricCallback):
    """
    Amount of correctly-classified fragments
    """
    def __init__(self, prefix: str, output_key: str, input_key: str = DataItemKeys.STRING):
        super().__init__(
            prefix=prefix,
            metric_fn=None,
            input_key=input_key,
            output_key=output_key)

        self.metrics = OcrMetrics()

    def on_loader_start(self, state):
        self.metrics.reset_counters()

    def on_batch_end(self, state):
        # Don't decode labels during training for training speedup
        if state.is_train_loader:
            return

        groundtruth = state.input[self.input_key]
        recognized = state.output[self.output_key]

        self.metrics.add_batch(recognized, groundtruth)

    def on_loader_end(self, state):
        # Don't decode labels during training for training speedup
        if state.is_train_loader:
            return

        values = self.metrics.get_metrics()
        for key, value in values.items():
            output_key = "{}.{}".format(self.prefix, key)
            state.loader_metrics[output_key] = value
