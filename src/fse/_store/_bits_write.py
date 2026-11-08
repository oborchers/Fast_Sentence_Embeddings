#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Writing the sign bits of one shard that is already published (docs/architecture/store.md,
"Sign bits").

The bits are derived from the mapped ``vectors`` of the shard in blocks, in row order. The
same bytes are hashed on the way and the hash must equal the checksum of the manifest of the
shard: bits made from a vector file that changed since the shard was written are never
published, because an untagged shard is not hashed when it is opened. The artifact is
written in ``.sign_bits-<first row>.writing-<hex>`` and renamed into place without replacing
anything, like a shard. A concurrent writer that was faster is not an error if its artifact
is valid.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _log, _reference_bits
from fse._container import (
    CorruptArtifactError,
    Hasher,
    NpyContainer,
    remove_temporary,
    tensor_code,
)
from fse._container._npy import write_manifest
from fse._container._npy_inplace import publish_directory

from ._bits_format import PREFIX, TENSOR, bits_name, metadata_of, open_bits, width
from ._kind import Shard
from ._layout import failed_cleanup

READ_BYTES = 1 << 24  # the vectors read for one block of rows


def created_by() -> str:
    try:
        return f"fse {version('fse')}"
    except PackageNotFoundError:  # pragma: no cover
        return "fse"


def derived(shard: Shard, vectors: ndarray) -> ndarray:
    """The bits of the rows of ``vectors`` (those of ``shard``). Raises CorruptArtifactError
    unless the bytes read are those whose checksum the manifest of the shard names."""
    dim = vectors.shape[1]
    out = np.empty((shard.rows, width(dim)), dtype=np.uint8)
    hasher = Hasher()
    step = max(1, READ_BYTES // max(1, dim * vectors.itemsize))
    for begin in range(0, shard.rows, step):
        block = np.array(
            vectors[begin : begin + step], copy=True, order="C"
        )  # hash and bits read these bytes
        hasher.update(block)
        out[begin : begin + step] = _reference_bits.sign_bits(block)
    if hasher.checksum() != shard.checksums["vectors"]:
        raise CorruptArtifactError(
            f"{shard.name}: the vectors do not match their checksum, so no sign bits were "
            "written; run fse.repair_store(path) while no writer or reader runs"
        )
    return out


def _write(tmp: Path, bits: ndarray, metadata: dict[str, str], *, synced: bool) -> None:
    """The artifact in the new directory ``tmp``: the file synced, the manifest renamed in
    last. The store directory is not synced here (``NpyContainer.write`` would): that is
    the business of the publication, and ``synced=False`` skips it."""
    tmp.mkdir()
    digest = NpyContainer._write_tensor(tmp / f"{TENSOR}.npy", bits)
    entries = {TENSOR: (tensor_code(bits), tuple(int(n) for n in bits.shape), digest)}
    write_manifest(
        tmp,
        entries,
        metadata,
        tmp / f".manifest.json.writing-{secrets.token_hex(4)}",
        directories=synced,
    )


def write_bits(
    path: Path, info: Any, shard: Shard, vectors: ndarray, *, synced: bool, by: str
) -> bool:
    """Derive and publish the bits of ``shard``; True if this call published them, False if
    another writer had published a valid artifact first. ``synced=False`` skips the sync of
    the store directory and tags the artifact ``fse.durability=none``."""
    return publish_bits(path, info, shard, derived(shard, vectors), synced=synced, by=by)


def publish_bits(
    path: Path, info: Any, shard: Shard, bits: ndarray, *, synced: bool, by: str
) -> bool:
    """Publish ``bits`` (those of the rows of ``shard``) as its artifact, like ``write_bits``
    but without deriving them: the caller made them from the rows it wrote (a sink) or checked
    them against the checksum of the shard (``derived``)."""
    tmp = path / f".{PREFIX}{shard.first_row:016d}.writing-{secrets.token_hex(4)}"
    try:
        _write(tmp, bits, metadata_of(info, shard, by, unsynced=not synced), synced=synced)
        publish_directory(tmp, path / bits_name(shard.first_row), sync_parent=synced)
    except FileExistsError:
        failed_cleanup(tmp, remove_temporary(tmp), None)
        found = open_bits(path, info, shard, verify=True)
        if found is None:  # pragma: no cover - it vanished
            raise
        if not np.array_equal(found[0], bits):  # an untagged artifact is not hashed on open
            raise CorruptArtifactError(
                f"{bits_name(shard.first_row)}: the artifact of another writer is not the bits "
                f"of {shard.name}; run fse.repair_store(path) while no writer or reader runs"
            ) from None
        return False
    except BaseException as error:
        failed_cleanup(tmp, remove_temporary(tmp), error)
        raise
    failed_cleanup(tmp, remove_temporary(tmp), None)  # gone once published
    return True


def write_all(
    root: Path, info: Any, todo: list, full: bool, workers: int, run_shards: Callable | None = None
) -> list[bool]:
    """Whether this call published the bits of each ``(shard, tensors)`` of ``todo``, in
    order; ``workers`` shards at once. A shard tagged ``fse.durability=none`` gets a tagged
    artifact, also with ``full``."""
    by = created_by()

    def one(item: tuple) -> bool:
        shard, tensors = item
        return write_bits(
            root, info, shard, tensors["vectors"], synced=full and not shard.unsynced, by=by
        )

    if run_shards is not None and workers > 1:
        return run_shards(one, todo)
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


def backfill(
    root: Path,
    info: Any,
    shards: list,
    full: bool,
    workers: int,
    start: float,
    run_shards: Callable | None = None,
) -> int:
    """Write the bits of every ``(shard, tensors)`` of ``shards`` that has none, log
    ``store.sign_bits_added`` and return the number of artifacts this call published."""
    todo = [(shard, tensors) for shard, tensors in shards if shard.sign_bits is None]
    written = write_all(root, info, todo, full, workers, run_shards)
    rows = sum(item[0].rows for item, done in zip(todo, written, strict=True) if done)
    _log.emit(
        "store.sign_bits_added",
        "fse store: wrote the sign bits of %d shards (%d rows)",
        sum(written),
        rows,
        shards=sum(written),
        rows=rows,
        bytes=rows * width(info.dim),
        durability="full" if full else "none",
        seconds=time.perf_counter() - start,
    )
    return sum(written)
