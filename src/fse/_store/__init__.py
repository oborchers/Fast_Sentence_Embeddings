#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Stores of sentence vectors: the ShardSink that writes them while an encoder runs and the
VectorStore that searches them (ARCHITECTURE.md, "sentence_vectors 2").

The layout is in ``_layout``, the writing of one shard in ``_shard``, the search by blocks
in ``_search``.
"""

from ._sink import ShardSink
from ._vectorstore import SearchResult, VectorStore

__all__ = ["SearchResult", "ShardSink", "VectorStore"]
