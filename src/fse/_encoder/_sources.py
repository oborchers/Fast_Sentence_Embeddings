#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""What ``encode`` and ``fit`` take as texts, and the source each becomes.

- a source of fse (StringSource, TokenLists, FileSource, StreamSource): as it is, token
  lists only for the external tokenizer;
- a path (``os.PathLike``): the lines of the file, opened and closed by the call;
- a list, tuple or numpy array of str: one sentence each; of lists of str: token lists,
  for the external tokenizer;
- any other iterable of str (a generator, a queue): a stream, read once.

A str is one sentence and never a path, so a bare str, bytes, sets and mappings raise
before anything is read.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from numpy import ndarray

from fse._contract import EXTERNAL_TOKENIZER
from fse._stream import StreamSource
from fse._text import StringSource, TokenLists
from fse._text_file import FileSource

SOURCES = (StringSource, TokenLists, FileSource, StreamSource)


@contextmanager
def opened(texts: Any, tokenizer: str) -> Iterator[Any]:
    """The source of ``texts`` for one call; a file the call opened is closed after it."""
    if isinstance(texts, SOURCES):
        if isinstance(texts, TokenLists):
            _check_token_lists(tokenizer)
        yield texts
        return
    if isinstance(texts, os.PathLike):
        with FileSource(texts) as source:
            yield source
        return
    _refuse(texts)
    if isinstance(texts, list | tuple | ndarray):
        yield _in_memory(texts, tokenizer)
        return
    yield StreamSource(texts)


def is_stream(source: Any) -> bool:
    """Whether ``source`` can be read only once."""
    return isinstance(source, StreamSource)


def _refuse(texts: Any):
    if isinstance(texts, str):
        raise TypeError(
            "texts must be a collection of sentences, not one str: pass [text] for one "
            "sentence, or a pathlib.Path for the lines of a file"
        )
    if isinstance(texts, bytes | bytearray | memoryview):
        raise TypeError(f"texts must be sentences as str, not {type(texts).__name__}")
    if isinstance(texts, set | frozenset | Mapping):
        raise TypeError(f"texts must be ordered, not a {type(texts).__name__}")


def _in_memory(texts: list | tuple | ndarray, tokenizer: str) -> StringSource | TokenLists:
    token_lists = len(texts) and isinstance(texts[0], list | tuple)
    if not token_lists:
        return StringSource(texts)
    _check_token_lists(tokenizer)
    return TokenLists(texts)


def _check_token_lists(tokenizer: str):
    if tokenizer != EXTERNAL_TOKENIZER:
        raise TypeError(
            f"lists of tokens are for the {EXTERNAL_TOKENIZER} tokenizer; this encoder "
            f"tokenizes with {tokenizer}: pass the sentences as str"
        )
