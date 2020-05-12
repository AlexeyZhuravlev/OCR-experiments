from catalyst.core import Callback, CallbackOrder

class DebugCallback(Callback):
    def __init__(self):
        super().__init__(order=CallbackOrder.Metric)

    def on_loader_start(self, state):
        for callback in state.callbacks:
            print(callback)
