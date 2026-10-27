import logging
from importlib.metadata import PackageNotFoundError, version

import gensim as _gensim

if int(_gensim.__version__.split(".")[0]) < 4:
    raise ImportError(
        f"fse requires gensim>=4.0, but gensim {_gensim.__version__} is installed. "
        "Upgrade gensim with: pip install -U 'gensim>=4'"
    )

from fse.inputs import (
    BaseIndexedList,
    CIndexedList,
    CorpusFile,
    CSplitCIndexedList,
    CSplitIndexedList,
    IndexedLineDocument,
    IndexedList,
    SplitCIndexedList,
    SplitIndexedList,
    TextList,
)
from fse.models import SIF, Average, SentenceVectors, uSIF
from fse.vectors import FTVectors, Vectors

try:
    __version__ = version("fse")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

logging.getLogger("fse").addHandler(logging.NullHandler())

__all__ = [
    "SIF",
    "Average",
    "BaseIndexedList",
    "CIndexedList",
    "CSplitCIndexedList",
    "CSplitIndexedList",
    "CorpusFile",
    "FTVectors",
    "IndexedLineDocument",
    "IndexedList",
    "SentenceVectors",
    "SplitCIndexedList",
    "SplitIndexedList",
    "TextList",
    "Vectors",
    "__version__",
    "uSIF",
]
