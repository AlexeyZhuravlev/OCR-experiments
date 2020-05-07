from .lstm_ctc import LstmCtcPredictionHead
from .factory import PredictionHeadsFactory

FACTORY = PredictionHeadsFactory()
FACTORY.register('lstm_ctc', LstmCtcPredictionHead)
