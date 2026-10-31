#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What ``encode`` takes as texts, and the source each becomes.

- a source of fse (StringSource, TokenListSource, FileSource, StreamSource): as it is,
  token lists only for the external tokenizer;
- a path (``os.PathLike``): the lines of the file, opened and closed by the call;
- a list, tuple or numpy array of str: one sentence each; of lists of str: token lists,
  for the external tokenizer;
- an Arrow string column (any object with ``__arrow_c_stream__`` or ``__arrow_c_array__``:
  a pyarrow Array or ChunkedArray, a pandas 3 Series, a Polars Series): its rows, read in
  place; if the producer cannot export (pandas without pyarrow), a stream with a warning;
- any other iterable of str (a generator, a queue): a stream, read once.

A str is one sentence and never a path, so a bare str, bytes, sets and mappings raise
before anything is read.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fse import _log
from fse._stream import StreamSource
from fse._text import ArrowSource, StringSource, TokenListSource, is_arrow
from fse._text_file import FileSource

from ._inputs import COLLECTIONS, build, check_tokenizer, is_token_lists, refuse

SOURCES = (StringSource, TokenListSource, FileSource, StreamSource, ArrowSource)


@contextmanager
def opened(texts: Any, tokenizer: str) -> Iterator[Any]:
    """The source of ``texts`` for one call; a file the call opened is closed after it."""
    if isinstance(texts, SOURCES):
        if isinstance(texts, TokenListSource):
            check_tokenizer(True, tokenizer)
        yield texts
        return
    if isinstance(texts, os.PathLike):
        with FileSource(texts) as source:
            yield source
        return
    if is_arrow(texts):
        arrow = _arrow_source(texts)
        if arrow is None:
            yield StreamSource(texts)
            return
        with arrow:
            yield arrow
        return
    refuse(texts)
    if isinstance(texts, COLLECTIONS):
        tokens = is_token_lists(texts)
        check_tokenizer(tokens, tokenizer)  # before any row is read
        yield build(texts, tokens)
        return
    yield StreamSource(texts)


def _arrow_source(texts: Any) -> ArrowSource | None:
    """The ArrowSource of ``texts``, or None (logged) where its producer cannot export
    Arrow data, such as a pandas Series of str without pyarrow: it is then read as a
    stream."""
    try:
        return ArrowSource(texts)
    except ImportError as error:
        _log.emit(
            "runtime.arrow_stream",
            "fse runtime: %s could not export Arrow data (%s); fse reads it as a stream of "
            "str, which is slower",
            type(texts).__name__,
            type(error).__name__,
            producer=type(texts).__name__,
            error=type(error).__name__,
        )
        return None
