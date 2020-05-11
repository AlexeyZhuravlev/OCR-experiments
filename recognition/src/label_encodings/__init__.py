from .ctc_encoding import CtcLabelEncoding
from .factory import LabelEncodingFactory

FACTORY = LabelEncodingFactory()

FACTORY.register('ctc', CtcLabelEncoding)
