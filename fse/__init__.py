from fse import models

from .inputs import BaseIndexedList
from .inputs import IndexedList
from .inputs import CIndexedList
from .inputs import SplitIndexedList
from .inputs import SplitCIndexedList
from .inputs import CSplitIndexedList
from .inputs import CSplitCIndexedList
from .inputs import IndexedLineDocument

import logging

class NullHandler(logging.Handler):
    def emit(self, record):
        pass

logger = logging.getLogger('fse')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())
