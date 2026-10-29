"""fse: fast sentence embeddings.

master is the development branch of fse 2.0: recipes (``Average``, ``SIF``, ``USIF``)
fit frozen ``Encoder`` objects on ``WordVectors``. The 1.x API (``uSIF``, the input
classes, ``Vectors``, the 1.x models) was removed here; it lives on in fse 1.1.0 on PyPI
and on the branch ``1.x``.
"""

import importlib as _importlib
import logging as _logging
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING
from typing import Any as _Any

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._wordvectors import WordVectors
from fse._wordvectors_format import FastTextParams

if TYPE_CHECKING:
    from fse._encoder import SIF, USIF, Average, Corpus, EncodedBatch, Encoder, QuantizedRows
    from fse._runtime import MemorySink, Sink
    from fse._store import SearchResult, ShardSink, VectorStore
del TYPE_CHECKING  # not a name of fse

# The encoder, the recipes and the sinks load on first access (PEP 562), not when fse is
# imported: the modules below them import fse._native, which import-linter cannot see as
# a module and counts as an import of fse itself, so a static import here would read as a
# cycle through every layer.
_LAZY = {
    **dict.fromkeys(
        ("SIF", "USIF", "Average", "Corpus", "Encoder", "EncodedBatch", "QuantizedRows"),
        "fse._encoder",
    ),
    **dict.fromkeys(("MemorySink", "Sink"), "fse._runtime"),
    **dict.fromkeys(("SearchResult", "ShardSink", "VectorStore"), "fse._store"),
}

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
    if name in _LAZY:
        value = getattr(_importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
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


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "SIF",
    "USIF",
    "Average",
    "Corpus",
    "CorruptArtifactError",
    "EncodedBatch",
    "Encoder",
    "FastTextParams",
    "MemorySink",
    "QuantizedRows",
    "SearchResult",
    "ShardSink",
    "Sink",
    "UnsupportedArtifactError",
    "VectorStore",
    "WordVectors",
    "__version__",
]
