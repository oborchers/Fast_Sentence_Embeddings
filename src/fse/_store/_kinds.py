#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The closed table of the kinds of rows a shard store holds (docs/architecture/store.md,
"Shared shard base").

``store.json`` of every store has the same four-part frame; which kind its metadata
belongs to decides how the root, the shards and the chain are read. The table is closed:
a kind is added here with its format, and nothing reads a kind by a name it finds in a file.
Today it holds ``sentence_vectors`` alone, and the kind of a root is always that one, so the
order of the checks of a root (the keys of the metadata before ``fse.kind``) is the order
it always was.
"""

from __future__ import annotations

from typing import Any

from ._kind import Kind
from ._vectors_format import VECTORS

KINDS: dict[str, Kind] = {VECTORS.name: VECTORS}


def kind_named(name: str) -> Kind:
    """The kind in the table under ``name``."""
    return KINDS[name]


def kind_of_root(metadata: Any) -> Kind:
    """The kind that reads the metadata of ``store.json``."""
    del metadata  # one kind: its parser names the keys it needs and refuses the rest
    return kind_named(VECTORS.name)


def kind_of(info: Any) -> Kind:
    """The kind of the store that ``info`` describes."""
    del info
    return kind_named(VECTORS.name)
