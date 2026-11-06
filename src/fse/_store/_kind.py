#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What the generic shard store knows of the kind of rows it holds (docs/architecture/store.md,
"Shared shard base").

A store is a directory of immutable shards named by their first row. What a shard holds
(tensors, metadata, the values that make it valid) is the business of its kind, a
``Kind``. The store layer opens shards, takes snapshots, verifies and repairs through this
protocol only, so a new kind of rows adds its format and nothing else.

Besides the checks of one shard, a kind may check the chain: ``summary`` is a small,
bounded value for each shard (read from the mapped tensors and the manifest, without
hashing), and ``follows`` says why a shard cannot follow its neighbour although both are
valid alone. Snapshot, ``verify`` and repair run the same chain, so what one of them
accepts the others accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from numpy import ndarray

from ._id_stats import IdStats

SHARD_PREFIX = "shard-"


def shard_name(first_row: int) -> str:
    return f"{SHARD_PREFIX}{first_row:016d}"


@dataclass(frozen=True)
class Shard:
    """One published shard: its rows ``first_row`` to ``first_row + rows - 1``."""

    name: str
    first_row: int
    rows: int
    unsynced: bool = False
    checksums: dict[str, str] = field(default_factory=dict)
    id_stats: IdStats | None = None
    # The fingerprint of its manifest, set when the shard is opened. ``refresh`` compares it
    # to see that the last shard of a snapshot is still the one on disk (not public).
    fingerprint: str = field(default="", kw_only=True, compare=False, repr=False)


class Kind(Protocol):
    """The format of the rows of a store: how its root and shards are read and checked."""

    name: str

    def parse_root(self, version: int, metadata: Any, file: Path) -> Any:
        """The info every row of the store shares, from the metadata of ``store.json``
        (``file``) of store version ``version``."""
        ...

    def open_shard(self, path: Path, info: Any, first_row: int) -> tuple[Shard, dict[str, ndarray]]:
        """The shard of the store ``path`` that starts at ``first_row``, mapped and checked
        against ``info``: its metadata, tensors and shapes."""
        ...

    def check_values(self, shard: Shard, tensors: dict[str, ndarray]) -> None:
        """Check the values of one shard in bounded blocks."""
        ...

    def check_identifiers(self, metadata: dict, file: Path, info: Any) -> None:
        """Recognize readable future identifiers in the metadata of a shard, before
        today's structure rules apply: UnsupportedArtifactError before any repair."""
        ...

    def summary(self, shard: Shard, tensors: dict[str, ndarray]) -> object:
        """What the chain needs of one shard: small and bounded."""
        ...

    def follows(self, previous: object, current: object) -> str | None:
        """Why the shard with the summary ``current`` cannot follow the one with
        ``previous``, or None."""
        ...
