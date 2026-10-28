"""fse: fast sentence embeddings.

master is the development branch of fse 2.0: recipes (``Average``, ``SIF``, ``USIF``)
fit frozen ``Encoder`` objects on ``WordVectors``. The 1.x API (``uSIF``, the input
classes, ``Vectors``, the 1.x models) was removed here; it lives on in fse 1.1.0 on PyPI
and on the branch ``1.x``.
"""

import logging as _logging
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import Any as _Any

from fse._encoder import SIF, USIF, Average, Encoder, QuantizedRows
from fse._wordvectors import WordVectors

_REMOVED = {
    "BaseIndexedList",
    "CIndexedList",
    "CorpusFile",
    "CSplitCIndexedList",
    "CSplitIndexedList",
    "FTVectors",
    "IndexedLineDocument",
    "IndexedList",
    "SentenceVectors",
    "SplitCIndexedList",
    "SplitIndexedList",
    "TextList",
    "Vectors",
    "uSIF",
}


def __getattr__(name: str) -> _Any:
    if name == "uSIF":
        raise AttributeError(
            "fse.uSIF is the 1.x model; the 2.0 recipe is fse.USIF (install fse==1.1.0 from "
            "PyPI or use the branch 1.x for the 1.x API)"
        )
    if name in _REMOVED:
        raise AttributeError(
            f"fse.{name} is part of the 1.x API, which the 2.0 development branch removed; "
            "install fse==1.1.0 from PyPI or use the branch 1.x"
        )
    raise AttributeError(f"module 'fse' has no attribute {name!r}")


try:
    __version__ = _version("fse")
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

_logging.getLogger("fse").addHandler(_logging.NullHandler())

__all__ = [
    "SIF",
    "USIF",
    "Average",
    "Encoder",
    "QuantizedRows",
    "WordVectors",
    "__version__",
]
