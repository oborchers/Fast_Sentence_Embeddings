import logging

from fse import models
from fse.models import SIF, Average, SentenceVectors, uSIF
from fse.vectors import FTVectors, Vectors

from .inputs import (
    BaseIndexedList,
    CIndexedList,
    CSplitCIndexedList,
    CSplitIndexedList,
    IndexedLineDocument,
    IndexedList,
    SplitCIndexedList,
    SplitIndexedList,
)


class NullHandler(logging.Handler):
    def emit(self, record):
        pass


logger = logging.getLogger("fse")
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())


__version__ = "0.2.0"
