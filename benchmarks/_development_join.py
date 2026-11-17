"""Text construction adapters for the existing local v2 controller."""

from __future__ import annotations

import os
from types import SimpleNamespace

CORPORA = ("join-short-400k", "join-unicode-400k")
STAGES = ("text-join", "token-join")
PROFILE = "text-join@1"


def validate(cell):
    if type(cell.get("join_spans", False)) is not bool:
        raise ValueError("join_spans must be boolean")
    if cell.get("join_spans") and (cell["stage"] != "text-join" or cell["workers"] not in (1, 8)):
        raise ValueError("join_spans requires text-join with one or eight requested parts")
    if (cell["stage"] in STAGES) != (cell["profile"] == PROFILE):
        raise ValueError("text-join and token-join require the text-join@1 profile exclusively")
    if cell["profile"] == PROFILE and cell["workers"] > 8:
        raise ValueError("text joins use at most eight native threads")


def corpus(name):
    if name == "join-short-400k":
        patterns = ["the quick brown fox jumps over the lazy dog " * 2 + "the end"]
    elif name == "join-unicode-400k":
        patterns = ["café déjà vu " * 4, "日本語 中文 한국어 " * 4, "music 𝄞 faces 😀 " * 4]
    else:
        raise ValueError("unknown text join corpus")
    texts = [patterns[i % len(patterns)] for i in range(400_000)]
    return dict(
        texts=texts,
        score_slices=[],
        source=dict(
            generator="text-join-corpora@1",
            patterns=patterns,
            rows=len(texts),
            character_lengths=[len(text) for text in patterns],
            utf8_lengths=[len(text.encode()) for text in patterns],
            order="cyclic patterns in declared order",
        ),
    )


def _arrays(source):
    import numpy as np

    return SimpleNamespace(
        tokens=np.frombuffer(source._data, dtype=np.uint8), offsets=source._offsets
    )


def operation(texts, job):
    from _development_encode import output_identity

    from fse._text import StringSource, TokenListSource
    from fse._text._strings import _encode_strings, _token_lists

    tokens = job["stage"] == "token-join"
    prepared = [text.split() for text in texts] if tokens else texts
    source_type = TokenListSource if tokens else StringSource
    # The public CPU override fixes the actual join thread budget in this child.
    os.environ["PYTHON_CPU_COUNT"] = str(job["workers"])
    expected_data, expected_offsets = (
        _token_lists(prepared) if tokens else _encode_strings(prepared, "string")
    )
    expected = output_identity(
        _arrays(SimpleNamespace(_data=expected_data, _offsets=expected_offsets))
    )
    del expected_data, expected_offsets

    def construct():
        return _arrays(source_type(prepared))

    if output_identity(construct()) != expected:
        raise ValueError("text construction bytes or offsets differ from Python reference")
    if job.get("join_spans", False):
        from _development_join_spans import observe

        construct.join_spans = lambda: observe(prepared, job, expected)
    return construct


def probe(model, texts, job):
    from _development_encode import output_identity

    proof = output_identity(operation(texts, job)())
    return dict(
        output=proof,
        source_rows=len(texts),
        encoder_fingerprint=model.fingerprint,
        dimensions=model.output_dim,
        reference="Python UTF-8 encoding, concatenation and cumulative byte lengths",
        valid_for_comparison=False,
        functional_only=True,
        artifact=job["artifact"],
        corpus=job["corpus"],
    )
