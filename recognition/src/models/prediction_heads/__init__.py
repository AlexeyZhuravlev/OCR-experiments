from .core import OcrPredictionHead, HeadOutputKeys
from .ctc import CtcPredictionHead
from .rnn_attention import RnnAttentionHead
from .factory import PredictionHeadsFactory

FACTORY = PredictionHeadsFactory()
FACTORY.register('ctc', CtcPredictionHead)
FACTORY.register('rnn_attn', RnnAttentionHead)

ATTENTION_HEAD_TYPES = set([
    'rnn_attn',
])
