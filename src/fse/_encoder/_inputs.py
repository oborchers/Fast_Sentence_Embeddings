#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The rules every in-memory input of ``encode`` follows.

A list, tuple or one-dimensional numpy array of str is a collection of texts; of lists
or tuples of str, a collection of token lists (for the external tokenizer). The first
row decides which, and a row of the other kind raises naming both rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from numpy import ndarray

from fse._contract import EXTERNAL_TOKENIZER
from fse._text import RowTypeError, StringSource, TokenListSource

COLLECTIONS = (list, tuple, ndarray)


def refuse(texts: Any):
    """Raise for what is no collection of texts, path or stream."""
    if isinstance(texts, str):
        raise TypeError(
            "texts must be a collection of sentences, not one str: pass [text] for one "
            "sentence, or a pathlib.Path for the lines of a file"
        )
    if isinstance(texts, bytes | bytearray | memoryview):
        raise TypeError(f"texts must be sentences as str, not {type(texts).__name__}")
    if isinstance(texts, set | frozenset | Mapping):
        raise TypeError(f"texts must be ordered, not a {type(texts).__name__}")
    kind = type(texts)
    if kind.__name__ == "Column" and kind.__module__.startswith("datasets."):
        raise TypeError(
            "a datasets Column exports no Arrow data: pass the column in Arrow format, "
            "dataset.with_format('arrow')['text'] (dataset.data.table['text'] ignores "
            "select and shuffle)"
        )
    if isinstance(texts, ndarray) and texts.ndim != 1:
        hint = ": pass arr.tolist() for a 2-D array of tokens" if texts.ndim == 2 else ""
        raise TypeError(f"a numpy array of texts must be one-dimensional, not {texts.ndim}-D{hint}")


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
    first = type(texts[0]).__name__ if len(texts) else None  # read before the rows
    try:
        return TokenListSource(texts) if tokens else StringSource(texts)
    except RowTypeError as error:
        # The error carries what it found, so nothing of the input is read again and no
        # error of user code can take its place.
        other = issubclass(error.found, str) if tokens else issubclass(error.found, list | tuple)
        if not other:
            raise TypeError(str(error)) from None
        raise TypeError(
            f"row {error.row} is a {error.found.__name__}, but row 0 is a {first}: texts "
            "are either str or token lists (lists or tuples of str), not both"
        ) from None
