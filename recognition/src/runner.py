from catalyst.dl import SupervisedRunner

class OcrRunner(SupervisedRunner):
    def __init__(self, model=None, device=None, input_key="image",
                 output_key="output", input_target_key="string"):
        super().__init__(model, device, input_key, output_key, input_target_key)
        