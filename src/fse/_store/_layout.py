#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The layout of a shard store (docs/architecture/store.md, "sentence_vectors 1").

A store is a directory: ``store.json``, written once when the store is created and
never changed, names what every row shares; the rows are immutable shards, each an
NpyContainer artifact ``shard-<first row>`` (at least 16 digits). A reader follows the
shards from row 0 by looking up the name of the next one, so a shard published while it
reads is either seen whole or not at all; entries starting with a dot (shards being
written, or left by a crash) and other names are ignored.

This module knows the frame of ``store.json`` and the names of the shards. What the
metadata and the shards of a store hold is the business of its kind (``_kind``,
``_kinds``). ``StoreInfo``, ``Shard``, ``shard_name`` and ``shard_metadata`` are
re-exported for the callers that name the layout.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

from numpy import ndarray

from fse import _log
from fse._container import (
    CorruptArtifactError,
    note_cleanup,
    remove_temporary,
)
from fse._container._common import _open_regular, _strict_json, _sync_directory
from fse._container._npy_inplace import publish_directory, sync_file, write_at
from fse._contract import UnsupportedArtifactError

from ._kind import SHARD_PREFIX, Shard, shard_name
from ._kinds import kind_of, kind_of_root
from ._vectors_format import StoreInfo, shard_metadata

__all__ = [
    "SHARD_PREFIX",
    "STORE_FILE",
    "STORE_NAME",
    "Shard",
    "StoreInfo",
    "create_store",
    "failed_cleanup",
    "open_shard",
    "read_store",
    "shard_metadata",
    "shard_name",
    "shards",
]

STORE_FILE = "store.json"
STORE_NAME = "fse_shards"
STORE_VERSION = 1
_STORE_BYTES = 1 << 16  # store.json is a few hundred bytes


def create_store(path: Path, info: StoreInfo) -> bool:
    """Create the store ``path`` with ``info`` unless it exists; True if this call made it.

    The store appears whole: ``store.json`` is written and synced in a temporary
    directory beside it, which is renamed into place without replacing anything, then
    the parent directory is synced. A store another writer created first is kept.
    """
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"the directory {path.parent} of the store {path.name} does not exist"
        )
    tmp = path.parent / f".{path.name}.creating-{secrets.token_hex(4)}"
    tmp.mkdir()
    try:
        text = json.dumps(
            {"store": STORE_NAME, "store_version": info.version, "metadata": info.metadata()},
            sort_keys=True,
            indent=1,
        )
        fd = os.open(
            tmp / STORE_FILE,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o644,
        )
        try:
            write_at(fd, memoryview((text + "\n").encode("utf-8")), 0)
            sync_file(fd)
        finally:
            os.close(fd)
        _sync_directory(tmp)  # the entry of store.json, before the store appears
        try:
            publish_directory(tmp, path)
            published = True
        except FileExistsError:
            published = False
    except BaseException as error:
        failed_cleanup(tmp, remove_temporary(tmp), error)
        raise
    failed_cleanup(tmp, remove_temporary(tmp), None)  # gone once published
    return published


def failed_cleanup(
    path: Path, failure: OSError | None, error: BaseException | None, *, failing: bool = False
) -> None:
    """Report ``failure``, the failed removal of the temporary ``path`` of a store (None:
    removed): raised unless an error is raised (``error``, or another one: ``failing``),
    else noted on ``error`` and logged, so it never replaces the error."""
    if failure is None:
        return
    if error is None and not failing:
        raise failure
    if error is not None:
        note_cleanup(error, path, failure)
    _log.emit(
        "store.cleanup_failed",
        "fse store: could not remove a temporary directory after an error (%s, errno %s); "
        "the error is raised with a note",
        type(failure).__name__,
        failure.errno,
        error=type(failure).__name__,
        errno=failure.errno,
    )


def read_store(path: Path) -> Any:
    """The info of the store ``path`` (what every row shares), checked strictly."""
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise CorruptArtifactError(f"{path} is no store of sentence vectors (no directory)")
    file = path / STORE_FILE
    with _open_regular(file) as handle:
        raw = handle.read(_STORE_BYTES + 1)
    if len(raw) > _STORE_BYTES:
        raise CorruptArtifactError(f"{file} is larger than {_STORE_BYTES} bytes")
    store = _strict_json(raw, file)
    if not isinstance(store, dict) or set(store) != {"store", "store_version", "metadata"}:
        raise CorruptArtifactError(f"{file} must hold exactly store, store_version and metadata")
    version = store["store_version"]
    if store["store"] != STORE_NAME or type(version) is not int or version not in (1, 2):
        raise UnsupportedArtifactError(
            f"{path} is a store {store['store']!r} version {version!r}; this fse "
            f"reads {STORE_NAME!r} version {STORE_VERSION}"
        )
    return kind_of_root(store["metadata"]).parse_root(version, store["metadata"], file)


def open_shard(path: Path, info: Any, first_row: int) -> tuple[Shard, dict[str, ndarray]]:
    """The shard of the store ``path`` that starts at ``first_row``, mapped and checked
    against the store: its metadata, tensors and shapes."""
    return kind_of(info).open_shard(path, info, first_row)


def shards(path: Path, info: Any) -> list[tuple[Shard, dict[str, ndarray]]]:
    """The published shards of the store, from row 0, each mapped: the next shard is
    looked up by name, so the rows always run without a gap; later shards are not read."""
    out = []
    row = 0
    while (path / shard_name(row)).exists() or (path / shard_name(row)).is_symlink():
        shard, tensors = open_shard(path, info, row)
        out.append((shard, tensors))
        row += shard.rows
    return out
