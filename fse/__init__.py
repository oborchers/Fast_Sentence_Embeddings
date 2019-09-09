from fse import models

from .inputs import BaseIndexedList, IndexedList, CIndexedList, SplitIndexedList, \
    CSplitIndexedList, CSplitCIndexedList, IndexedLineDocument

import logging

__version__ = '0.1'

class NullHandler(logging.Handler):
    def emit(self, record):
        pass


logger = logging.getLogger('fse')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())
