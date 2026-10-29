#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""What ``encode`` and ``fit`` take as texts, and the source each becomes.

- a ``fse.Corpus``, or a source of fse (StringSource, TokenListSource, FileSource,
  StreamSource): as it is, token lists only for the external tokenizer;
- a path (``os.PathLike``): the lines of the file, opened and closed by the call;
- a list, tuple or numpy array of str: one sentence each; of lists of str: token lists,
  for the external tokenizer;
- any other iterable of str (a generator, a queue): a stream, read once.

A str is one sentence and never a path, so a bare str, bytes, sets and mappings raise
before anything is read.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fse._stream import StreamSource
from fse._text import StringSource, TokenListSource
from fse._text_file import FileSource

from ._corpus import COLLECTIONS, Corpus, build, check_tokenizer, is_token_lists, refuse

SOURCES = (StringSource, TokenListSource, FileSource, StreamSource)


@contextmanager
def opened(texts: Any, tokenizer: str) -> Iterator[Any]:
    """The source of ``texts`` for one call; a file the call opened is closed after it."""
    if isinstance(texts, Corpus):
        check_tokenizer(texts._tokens, tokenizer)
        yield texts._source
        return
    if isinstance(texts, SOURCES):
        if isinstance(texts, TokenListSource):
            check_tokenizer(True, tokenizer)
        yield texts
        return
    if isinstance(texts, os.PathLike):
        with FileSource(texts) as source:
            yield source
        return
    refuse(texts)
    if isinstance(texts, COLLECTIONS):
        tokens = is_token_lists(texts)
        check_tokenizer(tokens, tokenizer)  # before any row is read
        yield build(texts, tokens)
        return
    yield StreamSource(texts)


def is_stream(source: Any) -> bool:
    """Whether ``source`` can be read only once."""
    return isinstance(source, StreamSource)
