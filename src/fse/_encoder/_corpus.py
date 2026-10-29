#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""``fse.Corpus``: texts in memory, read once, and the rules every in-memory input follows.

A list, tuple or one-dimensional numpy array of str is a collection of texts; of lists
or tuples of str, a collection of token lists (for the external tokenizer). The first
row decides which, and a row of the other kind raises naming both rows. ``fit`` and
``encode`` apply the same rules to a plain list; a ``Corpus`` applies them once.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from numpy import ndarray

from fse._contract import EXTERNAL_TOKENIZER
from fse._text import RowTypeError, StringSource, TokenListSource

COLLECTIONS = (list, tuple, ndarray)


class Corpus:
    """An immutable in-memory collection of texts or token sequences.

    ``fit`` and ``encode`` take lists as they are, but each call reads them again; a
    Corpus reads them once, so one object can be fitted on and encoded any number of
    times::

        corpus = fse.Corpus(texts)
        encoder = fse.SIF().fit(wv, corpus)
        rows = encoder.encode(corpus)

    ``texts`` is a list, tuple or one-dimensional numpy array, either of str (one text
    each) or of lists or tuples of str (the tokens of one text each, for encoders with
    ``tokenizer="external"``; tokens are non-empty and hold no whitespace). The first
    row decides which. The texts are copied, so later changes to ``texts`` do not change
    the corpus. A path (the lines of a file) and a stream (read once) are passed to fit
    and encode directly; ``encode_batches`` takes Corpus objects as batches.
    """

    __slots__ = ("_source", "_tokens")
    _source: StringSource | TokenListSource
    _tokens: bool

    def __init__(self, texts: Any):
        if hasattr(self, "_source"):  # __init__ called again on a made corpus
            raise AttributeError("a Corpus does not change; make a new one")
        if isinstance(texts, Corpus):  # shares the immutable source
            source, tokens = texts._source, texts._tokens
        else:
            refuse(texts, corpus=True)
            tokens = is_token_lists(texts)
            source = build(texts, tokens)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_tokens", tokens)

    def __setattr__(self, name: str, value: Any):
        raise AttributeError("a Corpus does not change; make a new one")

    def __delattr__(self, name: str):
        raise AttributeError("a Corpus does not change; make a new one")

    def __len__(self) -> int:
        return len(self._source)

    def __repr__(self) -> str:
        kind = "token lists" if self._tokens else "texts"
        return f"Corpus({len(self):,} {kind}, {self._source.nbytes:,} bytes)"


def refuse(texts: Any, *, corpus: bool = False):
    """Raise for what is not a collection of texts; a Corpus also refuses paths and
    streams, which fit and encode take directly."""
    if isinstance(texts, str):
        raise TypeError(
            "texts must be a collection of sentences, not one str: pass [text] for one "
            "sentence, or a pathlib.Path for the lines of a file"
        )
    if isinstance(texts, bytes | bytearray | memoryview):
        raise TypeError(f"texts must be sentences as str, not {type(texts).__name__}")
    if isinstance(texts, set | frozenset | Mapping):
        raise TypeError(f"texts must be ordered, not a {type(texts).__name__}")
    if isinstance(texts, ndarray) and texts.ndim != 1:
        hint = ": pass arr.tolist() for a 2-D array of tokens" if texts.ndim == 2 else ""
        raise TypeError(f"a numpy array of texts must be one-dimensional, not {texts.ndim}-D{hint}")
    if corpus and not isinstance(texts, COLLECTIONS):
        _refuse_for_corpus(texts)


def _refuse_for_corpus(texts: Any):
    if isinstance(texts, os.PathLike):
        raise TypeError(
            "a Corpus holds texts in memory; pass a path to fit or encode, which read the file"
        )
    raise TypeError(
        f"a Corpus is made from a list, tuple or numpy array, not a {type(texts).__name__}: "
        "pass a stream to fit or encode directly (it is read once), or list(...) it"
    )


def is_token_lists(texts: Any) -> bool:
    """Whether ``texts`` (a collection) holds token lists: its first row decides."""
    return bool(len(texts)) and isinstance(texts[0], list | tuple)


def check_tokenizer(tokens: bool, tokenizer: str):
    """Token lists need the external tokenizer; texts work with every tokenizer."""
    if tokens and tokenizer != EXTERNAL_TOKENIZER:
        raise TypeError(
            f"lists of tokens are for the {EXTERNAL_TOKENIZER} tokenizer; this encoder "
            f"tokenizes with {tokenizer}: pass the sentences as str"
        )


def build(texts: Any, tokens: bool) -> StringSource | TokenListSource:
    """The source of a collection; a row of the other kind than row 0 raises naming both."""
    try:
        return TokenListSource(texts) if tokens else StringSource(texts)
    except RowTypeError as error:
        # Only the row the builder named is read again, by index: no other row and no
        # method of a row runs, so no error of user code can take its place.
        row = texts[error.row]
        other = isinstance(row, str) if tokens else isinstance(row, list | tuple)
        if not other:
            raise
        raise TypeError(
            f"row {error.row} is a {type(row).__name__}, but row 0 is a "
            f"{type(texts[0]).__name__}: texts are either str or token lists (lists or tuples "
            "of str), not both"
        ) from None
