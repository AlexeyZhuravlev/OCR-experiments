from .ctc_encoding import CtcLabelEncoding
from .seq_encoding import SequenceLabelEncoding
from .factory import LabelEncodingFactory

FACTORY = LabelEncodingFactory()

FACTORY.register('ctc', CtcLabelEncoding)
FACTORY.register('seq', SequenceLabelEncoding)
