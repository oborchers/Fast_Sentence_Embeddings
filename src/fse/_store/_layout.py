#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The layout of a store of sentence vectors (ARCHITECTURE.md, "sentence_vectors 1").

A store is a directory: ``store.json``, written once when the store is created and
never changed, names the encoder, the storage dtype and the dimension; the rows are
immutable shards, each an NpyContainer artifact ``shard-<first row>`` (at least 16
digits) with ``vectors``, ``inv_norms`` and for int8 ``scales``. A reader follows the
shards from row 0 by looking up the name of the next one, so a shard published while it
reads is either seen whole or not at all; entries starting with a dot (shards being
written, or left by a crash) and other names are ignored.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _codes, _log
from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_FORM,
    FINGERPRINT_KEY,
    INTEGER,
    CorruptArtifactError,
    NpyContainer,
    is_fingerprint,
    note_cleanup,
    remove_temporary,
)
from fse._container._common import _open_regular, _strict_json, _sync_directory
from fse._container._npy_inplace import publish_directory, sync_file, write_at
from fse._contract import UnsupportedArtifactError

STORE_FILE = "store.json"
STORE_NAME = "fse_shards"
STORE_VERSION = 1
# The format of the sentence vectors; it starts with the xxh3-128 checksums (maintainer
# decision: the store was never released with sha256).
FORMAT_VERSION = "1"
SHARD_PREFIX = "shard-"
DTYPES = ("float32", "float16", "int8")
QUANTIZATION = "quantize_rows@1"
_STORE_BYTES = 1 << 16  # store.json is a few hundred bytes
# The metadata every shard shares with store.json.
_SHARED = (
    "fse.kind",
    "fse.format_version",
    "fse.transform_version",
    "fse.encoder",
    "fse.dtype",
    "fse.output_dim",
)


@dataclass(frozen=True)
class StoreInfo:
    """What every row of a store shares: the encoder (its fingerprint), the storage dtype
    and the dimension."""

    encoder: str
    dtype: str
    dim: int

    def metadata(self) -> dict[str, str]:
        return {
            "fse.kind": "sentence_vectors",
            "fse.format_version": FORMAT_VERSION,
            "fse.transform_version": "2",
            "fse.encoder": self.encoder,
            "fse.dtype": self.dtype,
            "fse.output_dim": str(self.dim),
        }

    def check(self, other: StoreInfo, path: Path) -> None:
        """Raise unless ``other`` describes the same rows as this store."""
        for name in ("encoder", "dtype", "dim"):
            if getattr(self, name) != getattr(other, name):
                raise ValueError(
                    f"the store {path} holds rows of {name} {getattr(self, name)!r}, not "
                    f"{getattr(other, name)!r}"
                )


@dataclass(frozen=True)
class Shard:
    """One published shard: its rows ``first_row`` to ``first_row + rows - 1``."""

    name: str
    first_row: int
    rows: int


def shard_name(first_row: int) -> str:
    return f"{SHARD_PREFIX}{first_row:016d}"


def shard_metadata(info: StoreInfo, first_row: int, rows: int, created_by: str) -> dict[str, str]:
    metadata = info.metadata()
    metadata[CREATED_BY_KEY] = created_by
    if info.dtype != "float32":
        metadata["fse.quantization"] = QUANTIZATION
    metadata["fse.first_row"] = str(first_row)
    metadata["fse.rows"] = str(rows)
    return metadata


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
            {"store": STORE_NAME, "store_version": STORE_VERSION, "metadata": info.metadata()},
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


def read_store(path: Path) -> StoreInfo:
    """The ``StoreInfo`` of the store ``path``, checked strictly."""
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
    if store["store"] != STORE_NAME or type(version) is not int or version != STORE_VERSION:
        raise UnsupportedArtifactError(
            f"{path} is a store {store['store']!r} version {version!r}; this fse "
            f"reads {STORE_NAME!r} version {STORE_VERSION}"
        )
    return _info(store["metadata"], file, ())


def _info(metadata: Any, file: Path, extra: tuple[str, ...]) -> StoreInfo:
    """The StoreInfo of ``metadata``, which must hold exactly the shared keys and
    ``extra``."""
    keys = (*_SHARED, *extra)
    if not isinstance(metadata, dict) or set(metadata) != set(keys):
        raise CorruptArtifactError(f"{file}: the metadata must hold exactly {', '.join(keys)}")
    if (
        metadata["fse.kind"] != "sentence_vectors"
        or metadata["fse.format_version"] != FORMAT_VERSION
        or metadata["fse.transform_version"] != "2"
    ):
        raise UnsupportedArtifactError(
            f"{file} holds {metadata['fse.kind']!r} format {metadata['fse.format_version']!r} "
            f"transform {metadata['fse.transform_version']!r}; this fse reads sentence_vectors "
            f"format {FORMAT_VERSION}, transform 2"
        )
    dim = metadata["fse.output_dim"]
    if (
        metadata["fse.dtype"] not in DTYPES
        or not isinstance(dim, str)
        or not INTEGER.fullmatch(dim)
        or dim == "0"
    ):
        raise CorruptArtifactError(f"{file}: fse.dtype or fse.output_dim is not valid")
    if not is_fingerprint(metadata["fse.encoder"]):
        raise CorruptArtifactError(
            f"{file}: fse.encoder must be an encoder fingerprint ({FINGERPRINT_FORM})"
        )
    return StoreInfo(metadata["fse.encoder"], metadata["fse.dtype"], int(dim))


def open_shard(path: Path, info: StoreInfo, first_row: int) -> tuple[Shard, dict[str, ndarray]]:
    """The shard of the store ``path`` that starts at ``first_row``, mapped and checked
    against the store: its metadata, tensors and shapes."""
    name = shard_name(first_row)
    tensors, metadata = NpyContainer().open(path / name, mmap=True)
    file = path / name / "manifest.json"
    content = {  # without the checksums and the fingerprint the container checked
        k: v
        for k, v in metadata.items()
        if k != FINGERPRINT_KEY and not k.startswith(CHECKSUM_PREFIX)
    }
    shard_info = _info(content, file, _shard_keys(info))
    info.check(shard_info, path)
    _check_quantization(metadata, info, file)
    rows = metadata.get("fse.rows", "")
    if (
        metadata.get("fse.first_row") != str(first_row)
        or not INTEGER.fullmatch(rows)
        or rows == "0"
    ):
        raise CorruptArtifactError(f"{file}: the shard must hold rows from {first_row}")
    want = {"vectors", "inv_norms"} | ({"scales"} if info.dtype == "int8" else set())
    if set(tensors) != want:
        raise CorruptArtifactError(f"{file}: a {info.dtype} shard holds exactly {sorted(want)}")
    n = int(rows)
    if tensors["vectors"].shape != (n, info.dim) or any(
        tensors[k].shape != (n,) for k in want - {"vectors"}
    ):
        raise CorruptArtifactError(f"{file}: the tensors do not hold {n} rows of {info.dim} values")
    if tensors["vectors"].dtype != np.dtype(_codes.STORAGE_TYPES[info.dtype]) or any(
        tensors[k].dtype != np.float32 for k in want - {"vectors"}
    ):
        raise CorruptArtifactError(
            f"{file}: the vectors must be {info.dtype}, the inverse norms and scales float32"
        )
    return Shard(name, first_row, n), tensors


def _shard_keys(info: StoreInfo) -> tuple[str, ...]:
    """The keys a shard adds to the shared metadata."""
    keys = (CREATED_BY_KEY, "fse.first_row", "fse.rows")
    return keys if info.dtype == "float32" else (*keys, "fse.quantization")


def _check_quantization(metadata: dict, info: StoreInfo, file: Path) -> None:
    """A float16 or int8 shard names the storage rule of its codes: quantize_rows@1."""
    rule = metadata.get("fse.quantization")
    if info.dtype == "float32" or rule == QUANTIZATION:
        return
    if isinstance(rule, str) and rule.startswith("quantize_rows@"):
        raise UnsupportedArtifactError(f"{file} is stored by {rule}; this fse reads {QUANTIZATION}")
    raise CorruptArtifactError(f"{file}: fse.quantization must be {QUANTIZATION}")


def shards(path: Path, info: StoreInfo) -> list[tuple[Shard, dict[str, ndarray]]]:
    """The published shards of the store, from row 0, each mapped: the next shard is
    looked up by name, so the rows always run without a gap; later shards are not read."""
    out = []
    row = 0
    while (path / shard_name(row)).exists() or (path / shard_name(row)).is_symlink():
        shard, tensors = open_shard(path, info, row)
        out.append((shard, tensors))
        row += shard.rows
    return out
