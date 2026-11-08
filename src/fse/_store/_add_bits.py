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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fse import _log
from fse._runtime import workers_of
from fse._validation import PathLike, PositiveInt, validate_arguments

from ._bits_format import width
from ._bits_write import created_by, write_bits
from ._layout import read_store
from ._recovery import clear_exception_frames, snapshot
from ._vectors_format import VECTORS


@validate_arguments
def add_sign_bits(
    path: PathLike,
    *,
    durability: Literal["full", "none"] = "full",
    workers: PositiveInt | None = None,
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
    ``UnsupportedArtifactError``."""
    start = time.perf_counter()
    root = Path(path)
    info = read_store(root, VECTORS.name)
    count = workers_of(workers).workers
    try:
        checked = snapshot(root, info, verify=True)
        try:
            todo = [
                (shard, tensors) for shard, tensors in checked.shards if shard.sign_bits is None
            ]
            written = _write_all(root, info, todo, durability == "full", count)
        finally:
            checked.shards.clear()  # release the mappings
    except BaseException as error:
        clear_exception_frames(error)
        raise
    rows = sum(item[0].rows for item, done in zip(todo, written, strict=True) if done)
    _log.emit(
        "store.sign_bits_added",
        "fse store: wrote the sign bits of %d shards (%d rows)",
        sum(written),
        rows,
        shards=sum(written),
        rows=rows,
        bytes=rows * width(info.dim),
        durability=durability,
        seconds=time.perf_counter() - start,
    )
    return sum(written)


def _write_all(root: Path, info: Any, todo: list, full: bool, workers: int) -> list[bool]:
    """Whether this call published the bits of each shard of ``todo``, in order."""
    by = created_by()

    def one(item: tuple) -> bool:
        shard, tensors = item
        return write_bits(
            root, info, shard, tensors["vectors"], synced=full and not shard.unsynced, by=by
        )

    if workers == 1 or len(todo) <= 1:
        return [one(item) for item in todo]
    with ThreadPoolExecutor(min(workers, len(todo)), thread_name_prefix="fse-sign-bits") as pool:
        futures = [pool.submit(one, item) for item in todo]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
