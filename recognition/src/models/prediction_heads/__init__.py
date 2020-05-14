from .core import OcrPredictionHead
from .lstm_ctc import LstmCtcPredictionHead
from .rnn_attention import RnnAttentionHead
from .factory import PredictionHeadsFactory

FACTORY = PredictionHeadsFactory()
FACTORY.register('lstm_ctc', LstmCtcPredictionHead)
FACTORY.register('rnn_attn', RnnAttentionHead)
