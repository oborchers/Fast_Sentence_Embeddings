#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Text sources: sentences of text to batches, stages 1 and 2 (ARCHITECTURE.md, "Tokenizers").

A source holds its sentences as one UTF-8 buffer: strings in memory with their offsets
(``StringSource``, ``TokenListSource``), or the lines of a memory-mapped file
(``fse._text_file.FileSource``). ``scan`` validates and counts them in threads without the GIL and cuts
them into chunks; ``encode`` turns one chunk into a Batch with a ``TextLookup`` and a
per-worker ``Workspace``. Every sentence is one row, empty ones included.
"""

from ._arrow import ArrowSource, is_arrow
from ._lookup import (
    MAX_CHUNK_ROWS,
    MAX_NGRAMS_LIMIT,
    TextLookup,
    Vocabulary,
    Workspace,
    make_lookup,
    make_vocabulary,
    new_workspace,
    tokenize,
)
from ._scan import Chunk, Scan, _Buffered, check_chunk_bytes, default_chunk_bytes
from ._strings import RowTypeError, StringSource, TokenListSource, encode_string

__all__ = [
    "MAX_CHUNK_ROWS",
    "MAX_NGRAMS_LIMIT",
    "ArrowSource",
    "Chunk",
    "RowTypeError",
    "Scan",
    "StringSource",
    "TextLookup",
    "TokenListSource",
    "Vocabulary",
    "Workspace",
    "_Buffered",
    "check_chunk_bytes",
    "default_chunk_bytes",
    "encode_string",
    "is_arrow",
    "make_lookup",
    "make_vocabulary",
    "new_workspace",
    "tokenize",
]
