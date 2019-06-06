from fse import models, exp
import logging

__version__ = '0.1'

class NullHandler(logging.Handler):
    def emit(self, record):
        pass


logger = logging.getLogger('fse')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())
