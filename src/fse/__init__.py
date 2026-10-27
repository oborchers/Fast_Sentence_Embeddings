"""fse: fast sentence embeddings.

The 2.0 classes import without gensim. The 1.x classes (models, inputs and the hub
vectors) are loaded on first access and need gensim 4 until 2.0 replaces them.
"""

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fse._wordvectors import WordVectors

# The 1.x names and their modules, loaded on first access because they import gensim.
_LEGACY = {
    "BaseIndexedList": "fse.inputs",
    "CIndexedList": "fse.inputs",
    "CorpusFile": "fse.inputs",
    "CSplitCIndexedList": "fse.inputs",
    "CSplitIndexedList": "fse.inputs",
    "IndexedLineDocument": "fse.inputs",
    "IndexedList": "fse.inputs",
    "SplitCIndexedList": "fse.inputs",
    "SplitIndexedList": "fse.inputs",
    "TextList": "fse.inputs",
    "SIF": "fse.models",
    "Average": "fse.models",
    "SentenceVectors": "fse.models",
    "uSIF": "fse.models",
    "FTVectors": "fse.vectors",
    "Vectors": "fse.vectors",
}


def _require_gensim_4():
    import gensim  # noqa: PLC0415 - only the 1.x classes need it

    if int(gensim.__version__.split(".")[0]) < 4:
        raise ImportError(
            f"fse requires gensim>=4.0, but gensim {gensim.__version__} is installed. "
            "Upgrade gensim with: pip install -U 'gensim>=4'"
        )


def __getattr__(name: str) -> Any:
    module = _LEGACY.get(name)
    if module is None:
        raise AttributeError(f"module 'fse' has no attribute {name!r}")
    _require_gensim_4()
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LEGACY))


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
    "WordVectors",
    "__version__",
    "uSIF",
]
