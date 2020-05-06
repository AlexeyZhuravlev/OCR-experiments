from catalyst.dl import SupervisedRunner

class OcrRunner(SupervisedRunner):
    def __init__(self, model=None, device=None, input_key="image",
                 output_key="logits", input_target_key="targets"):
        super().__init__(model, device, input_key, output_key, input_target_key)
