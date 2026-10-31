#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Sentences as str and as token lists in memory, joined into one UTF-8 buffer by one
native pass (``_native_lists.pxi``) or, for the inputs it does not read, in Python."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from fse import _native
from fse._contract import EXTERNAL_TOKENIZER, Batch
from fse._cpus import usable_cpus

from ._scan import Chunk, _Buffered


def _check_container(texts: object, what: str) -> None:
    if isinstance(texts, set | frozenset):
        raise TypeError(f"{what} must be ordered, not a set; pass sorted(...) instead")
    if not isinstance(texts, list | tuple | ndarray):
        raise TypeError(f"{what} must be a list, tuple or numpy array, not {type(texts).__name__}")


class RowTypeError(TypeError):
    """A row of a collection is not of the type its kind needs (encode names both rows
    when it is of the other kind): ``row`` is its index, ``found`` its type."""

    def __init__(self, message: str, row: int, found: type):
        super().__init__(message)
        self.row = row
        self.found = found


def encode_string(text: object, what: str, row: int) -> bytes:
    """The UTF-8 bytes of sentence ``row``; a non-str or a lone surrogate raises."""
    if not isinstance(text, str):
        raise RowTypeError(f"{what} {row} is a {type(text).__name__}, not a str", row, type(text))
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as err:
        raise UnicodeEncodeError(
            err.encoding, err.object, err.start, err.end, f"{err.reason} in {what} {row}"
        ) from None


def _join(encoded: list[bytes]) -> tuple[bytes, ndarray]:
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    return b"".join(encoded), offsets


def _encode_strings(texts: Any, what: str) -> tuple[bytes, ndarray]:
    return _join([encode_string(text, what, i) for i, text in enumerate(texts)])


# Threads that join strings or token lists: the pass is bound by reading memory, which a
# few threads saturate (85.7 million WikiText tokens on an M1 Pro, in a fresh process:
# 2.1 s with one thread, 0.65 s with 8).
_JOIN_THREADS = 8


class StringSource(_Buffered):
    """Sentences as strings in memory, one row each."""

    def __init__(self, texts: Any):
        _check_container(texts, "texts")
        joined = _native.join_strings(texts, min(_JOIN_THREADS, usable_cpus()))
        # None: types the native pass does not read (a str subclass may override encode,
        # an array) or a lone surrogate; the Python path decides and raises the error.
        if joined is None:
            joined = _encode_strings(texts, "string")
        else:
            self._verify(*joined)
        self._data, self._offsets = joined

    @classmethod
    def encoded(cls, sentences: list[bytes]) -> StringSource:
        """A source of sentences already encoded (and checked) as UTF-8."""
        source = cls.__new__(cls)
        source._data, source._offsets = _join(sentences)
        return source

    def __len__(self) -> int:
        return self._positions()


def _token_lists(sentences: Any) -> tuple[bytes, ndarray]:
    """The joined token lists in Python, for the inputs the native pass does not take and
    for the errors, which it leaves to this path (types first, then tokens, then the UTF-8
    of every sentence)."""
    joined = []
    for i, tokens in enumerate(sentences):
        if not isinstance(tokens, list | tuple):
            message = f"sentence {i} is a {type(tokens).__name__}, not a list of str"
            raise RowTypeError(message, i, type(tokens))
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError(f"sentence {i} holds a {type(token).__name__}, not a str")
            if not token or token.split() != [token]:
                raise ValueError(
                    f"sentence {i} holds the token {token!r}: tokens are non-empty and hold no whitespace"
                )
        joined.append(" ".join(tokens))
    return _encode_strings(joined, "sentence")


class TokenListSource(_Buffered):
    """Sentences given as lists of tokens, for the external tokenizer only. Tokens must be non-empty and hold no whitespace, so splitting the joined
    sentence at whitespace gives them back exactly.
    """

    def __init__(self, sentences: Any):
        _check_container(sentences, "sentences")
        joined = _native.join_token_lists(sentences, min(_JOIN_THREADS, usable_cpus()))
        # None: types the native pass does not read (subclasses, an array); False: an
        # invalid token. The Python path decides both: it raises the error, or succeeds
        # where a subclass changed the lists while they were read.
        if joined is None or joined is False:
            joined = _token_lists(sentences)
        else:
            self._verify(*joined)
        self._data, self._offsets = joined

    def __len__(self) -> int:
        return self._positions()

    def encode(
        self, chunk: Chunk, lookup: _native.TextLookup, workspace: _native.Workspace
    ) -> Batch:
        if lookup.tokenizer != _native.TOKENIZER_EXTERNAL:
            raise ValueError(f"token lists are for the {EXTERNAL_TOKENIZER} tokenizer")
        return super().encode(chunk, lookup, workspace)
