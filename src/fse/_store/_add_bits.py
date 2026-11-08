#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.add_sign_bits``: the sign bits for the shards of an existing store that have none
(docs/architecture/store.md, "Sign bits").

A store written without bits, or a shard appended without them, gets the companion artifact of
each shard later and without touching the shard. The store is opened and checked as
``VectorStore.open`` does; every shard without bits is read in blocks, its bytes are hashed
against the checksum of its manifest while the bits are derived, and the artifact is published
only if they match (``_bits_write``). A rerun after an interruption does the shards that are
left; what was published stays and is valid.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from fse._runtime import Pool, workers_of
from fse._validation import PathLike, PositiveInt, validate_arguments

from . import _search_run
from ._bits_write import backfill
from ._layout import read_store
from ._recovery import clear_exception_frames, snapshot
from ._vectors_format import VECTORS


@validate_arguments
def add_sign_bits(
    path: PathLike,
    *,
    durability: Literal["full", "none"] = "full",
    workers: PositiveInt | None = None,
    pool: Pool | None = None,
) -> int:
    """Write the sign bits of every shard of the store ``path`` that has none and return the
    number of artifacts written (0 if every shard has them).

    The store is checked as ``VectorStore.open`` checks it, so damage raises
    ``CorruptArtifactError`` before anything is written; a shard whose vectors no longer match
    their checksum raises it too, and nothing is published for it. ``durability="none"``
    skips the sync of the store directory and tags the artifacts (which are then hashed when
    the store is opened); the artifact of a shard written with ``durability="none"`` is tagged
    in any case. ``workers`` shards are done at once (default: the usable CPUs). Run it while
    writers append if you must: a shard published meanwhile is not covered, and an artifact
    another writer published first is kept when it is valid. A score store raises
    ``UnsupportedArtifactError``. With ``pool=``, shards share its threads and task
    places. ``workers`` defaults to its size and must not exceed it. The caller owns
    the pool; inline work (``workers=1``) takes no place."""
    start = time.perf_counter()
    root = Path(path)
    threads = workers_of(workers, pool)
    info = read_store(root, VECTORS.name)

    def run_shards(fn, items):
        return list(_search_run.run(fn, items, threads, "fse-sign-bits"))
    try:
        checked = snapshot(root, info, verify=True)
        try:
            return backfill(
                root, info, checked.shards, durability == "full", threads.workers, start,
                run_shards=run_shards if pool is not None else None,
            )
        finally:
            checked.shards.clear()  # release the mappings
    except BaseException as error:
        clear_exception_frames(error)
        raise
