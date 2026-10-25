import logging
from importlib.metadata import PackageNotFoundError, version

import gensim

if int(gensim.__version__.split(".")[0]) < 4:
    raise ImportError(
        f"fse requires gensim>=4.0, but gensim {gensim.__version__} is installed. "
        "Upgrade gensim with: pip install -U 'gensim>=4'"
    )

from fse.inputs import (  # noqa: E402
    BaseIndexedList,
    CIndexedList,
    CSplitCIndexedList,
    CSplitIndexedList,
    IndexedLineDocument,
    IndexedList,
    SplitCIndexedList,
    SplitIndexedList,
)
from fse.models import SIF, Average, SentenceVectors, uSIF  # noqa: E402
from fse.vectors import FTVectors, Vectors  # noqa: E402

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
    "FTVectors",
    "IndexedLineDocument",
    "IndexedList",
    "SentenceVectors",
    "SplitCIndexedList",
    "SplitIndexedList",
    "Vectors",
    "__version__",
    "uSIF",
]
