"""Types that fse returns or accepts, for annotating caller code.

``ScoreHead``, ``Tokenizer`` and ``Encoding`` describe returned objects rather than
objects users construct. ``SinkRows`` is what ``Sink.target`` returns and ``commit``
accepts. ``LengthLevel`` selects a length report. ``PathLike`` names local text paths;
``ModelRef`` also permits string Hub identifiers. ``WorkerCount`` is a positive Python
or NumPy integer, or None for the available CPUs. Importing this module loads the
native extension; annotations can import it under ``typing.TYPE_CHECKING``.
"""

from fse._runtime._lengths import LengthLevel
from fse._runtime._sinks import Stored as SinkRows
from fse._static_model._tokenizer import Encoding, Tokenizer
from fse._store._scores_info import ScoreHead
from fse._validation import ModelRef, PathLike, WorkerCount

__all__ = [
    "Encoding",
    "LengthLevel",
    "ModelRef",
    "PathLike",
    "ScoreHead",
    "SinkRows",
    "Tokenizer",
    "WorkerCount",
]
