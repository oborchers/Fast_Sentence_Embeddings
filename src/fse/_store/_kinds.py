#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The closed table of the kinds of rows a shard store holds (docs/architecture/store.md,
"Shared shard base").

``store.json`` of every store has the same four-part frame; which kind its metadata
belongs to decides how the root, the shards and the chain are read. The table is closed:
a kind is added here with its format, and nothing reads a kind by a name it finds in a file.
It holds ``sentence_vectors`` and ``prediction_scores``. A root that names no kind (a missing
or non-text ``fse.kind``) is read by ``sentence_vectors``, so its checks (the keys of the
metadata before ``fse.kind``) and messages are the ones it always had; a text that names no
kind of the table is an ``UnsupportedArtifactError``.
"""

from __future__ import annotations

from typing import Any

from fse._contract import UnsupportedArtifactError

from ._kind import Kind
from ._scores_format import SCORES
from ._vectors_format import VECTORS

KINDS: dict[str, Kind] = {VECTORS.name: VECTORS, SCORES.name: SCORES}


def kind_named(name: str) -> Kind:
    """The kind in the table under ``name``."""
    return KINDS[name]


def kind_of_root(metadata: Any) -> Kind:
    """The kind that reads the metadata of ``store.json``: the one ``fse.kind`` names."""
    name = metadata.get("fse.kind") if isinstance(metadata, dict) else None
    if not isinstance(name, str):
        return kind_named(VECTORS.name)  # its parser names the keys it needs and refuses the rest
    if name not in KINDS:
        raise UnsupportedArtifactError(
            f"this store holds the kind {name!r}; this fse reads {', '.join(sorted(KINDS))}"
        )
    return kind_named(name)


def kind_of(info: Any) -> Kind:
    """The kind of the store that ``info`` describes."""
    return kind_named(info.kind)
