"""The public typing module exposes the original objects and no implementation helpers."""

from typing import get_args

import fse
import fse.typing as public
from fse._runtime._lengths import LengthLevel
from fse._runtime._sinks import Stored
from fse._static_model._tokenizer import Encoding, Tokenizer
from fse._store._scores_info import ScoreHead
from fse._validation import ModelRef, PathLike, WorkerCount


def test_public_type_exports_are_the_original_objects():
    assert public.__all__ == [
        "Encoding",
        "LengthLevel",
        "ModelRef",
        "PathLike",
        "ScoreHead",
        "SinkRows",
        "Tokenizer",
        "WorkerCount",
    ]
    assert public.Encoding is Encoding
    assert public.LengthLevel is LengthLevel
    assert public.ScoreHead is ScoreHead
    assert public.SinkRows is Stored
    assert public.Tokenizer is Tokenizer
    assert public.PathLike is PathLike
    assert public.ModelRef is ModelRef
    assert public.WorkerCount is WorkerCount
    assert not hasattr(public, "TextInput")
    assert not hasattr(public, "TextBatches")
    assert get_args(public.LengthLevel) == ("summary", "per_row")
    assert not set(public.__all__) & set(fse.__all__)
    assert "typing" not in fse.__all__
    assert fse.typing is public
    assert not hasattr(public, "ScoreSchema")
    assert not hasattr(ScoreHead, "entry")
    assert not hasattr(ScoreHead, "threshold_name")
