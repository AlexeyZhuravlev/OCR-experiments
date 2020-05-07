from catalyst.dl import registry

from runner import OcrRunner as Runner
from experiment import OcrExperiment as Experiment
from catalyst.dl import registry
from . import models

registry.MODELS.add_from_module(models)
