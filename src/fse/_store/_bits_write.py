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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _reference_bits
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
    bits = derived(shard, vectors)
    tmp = path / f".{PREFIX}{shard.first_row:016d}.writing-{secrets.token_hex(4)}"
    try:
        _write(tmp, bits, metadata_of(info, shard, by, unsynced=not synced), synced=synced)
        publish_directory(tmp, path / bits_name(shard.first_row), sync_parent=synced)
    except FileExistsError:
        failed_cleanup(tmp, remove_temporary(tmp), None)
        if open_bits(path, info, shard, verify=True) is None:  # pragma: no cover - it vanished
            raise
        return False
    except BaseException as error:
        failed_cleanup(tmp, remove_temporary(tmp), error)
        raise
    failed_cleanup(tmp, remove_temporary(tmp), None)  # gone once published
    return True
