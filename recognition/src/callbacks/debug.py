from catalyst.core import Callback, CallbackOrder

class DebugCallback(Callback):
    def __init__(self):
        super().__init__(order=CallbackOrder.Metric)

    def on_batch_end(self, state):
        print("Output")
        print(state.output)
        print("Input")
        print(state.input)
