from catalyst.dl import registry

from .runner import OcrRunner as Runner
from .experiment import OcrExperiment as Experiment
from . import models
from . import callbacks

registry.CALLBACKS.add_from_module(callbacks)
registry.MODELS.add_from_module(models)
