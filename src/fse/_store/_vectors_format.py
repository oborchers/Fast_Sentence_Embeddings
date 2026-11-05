#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The kind ``sentence_vectors`` (docs/architecture/store.md, "sentence_vectors 1").

``store.json`` names the encoder, the storage dtype and the dimension; the rows are shards,
each an NpyContainer artifact with ``vectors``, ``inv_norms``, for int8 ``scales`` and with
document IDs ``doc_ids``. This module holds the schema of the root and of a shard, the
checks of one shard and the kind the store layer opens it by. The chain check has nothing to
say about vectors: every shard follows every other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_FORM,
    FINGERPRINT_KEY,
    INTEGER,
    CorruptArtifactError,
    NpyContainer,
    is_fingerprint,
)
from fse._contract import UnsupportedArtifactError, parse_id

from ._id_stats import (
    ID_KEYS,
    MODE_KEYS,
    check_mode,
    check_removed,
    parsed_stats,
    verify_stats,
)
from ._kind import Shard, shard_name

NAME = "sentence_vectors"
# The format of the sentence vectors; it starts with the xxh3-128 checksums (maintainer
# decision: the store was never released with sha256).
FORMAT_VERSION = "1"
DTYPES = ("float32", "float16", "int8")
QUANTIZATION = "quantize_rows@1"
_VALUE_BYTES = 1 << 20
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
    has_doc_ids: bool = False
    version: int = 1

    @property
    def kind(self) -> str:
        return NAME

    def metadata(self) -> dict[str, str]:
        metadata = {
            "fse.kind": NAME,
            "fse.format_version": str(self.version),
            "fse.transform_version": "2",
            "fse.encoder": self.encoder,
            "fse.dtype": self.dtype,
            "fse.output_dim": str(self.dim),
        }

        if self.version == 2:
            metadata.update(
                {"fse.doc_ids": "int64" if self.has_doc_ids else "none", "fse.resume": "none"}
            )
        return metadata

    def check(self, other: StoreInfo, path: Path) -> None:
        """Raise unless ``other`` describes the same rows as this store."""
        for name in ("encoder", "dtype", "dim", "has_doc_ids"):
            if getattr(self, name) != getattr(other, name):
                raise ValueError(
                    f"the store {path} holds rows of {name} {getattr(self, name)!r}, not "
                    f"{getattr(other, name)!r}"
                )


def shard_metadata(
    info: StoreInfo, first_row: int, rows: int, created_by: str, *, unsynced: bool = False
) -> dict[str, str]:
    metadata = info.metadata()
    metadata[CREATED_BY_KEY] = created_by
    if info.dtype != "float32":
        metadata["fse.quantization"] = QUANTIZATION
    metadata["fse.first_row"] = str(first_row)
    metadata["fse.rows"] = str(rows)
    if unsynced:
        metadata["fse.durability"] = "none"
    return metadata


def _info(metadata: Any, file: Path, extra: tuple[str, ...]) -> StoreInfo:
    """The StoreInfo of ``metadata``, which must hold exactly the shared keys and
    ``extra``."""
    if not isinstance(metadata, dict):
        raise CorruptArtifactError(f"{file}: metadata must be an object")
    version = metadata.get("fse.format_version")
    check_mode(metadata, file)
    if isinstance(version, str) and version not in ("1", "2"):
        raise UnsupportedArtifactError(f"{file}: unsupported sentence_vectors format {version!r}")
    keys = (*_SHARED, *(MODE_KEYS if version == "2" else ()), *extra)
    if not isinstance(metadata, dict) or set(metadata) != set(keys):
        raise CorruptArtifactError(f"{file}: the metadata must hold exactly {', '.join(keys)}")
    if (
        metadata["fse.kind"] != NAME
        or metadata["fse.format_version"] not in ("1", "2")
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
    if version == "2" and any(
        metadata.get(key) not in values
        for key, values in (("fse.doc_ids", ("none", "int64")), ("fse.resume", ("none",)))
    ):
        raise CorruptArtifactError(f"{file}: invalid document-ID or resume mode")
    return StoreInfo(
        metadata["fse.encoder"],
        metadata["fse.dtype"],
        int(dim),
        metadata.get("fse.doc_ids") == "int64",
        int(str(version)),
    )


def open_shard(path: Path, info: StoreInfo, first_row: int) -> tuple[Shard, dict[str, ndarray]]:
    """The shard of the store ``path`` that starts at ``first_row``, mapped and checked
    against the store: its metadata, tensors and shapes."""
    name = shard_name(first_row)
    directory = path / name
    if not directory.is_dir() or directory.is_symlink():
        raise CorruptArtifactError(f"{directory} is not a regular shard directory")
    file = path / name / "manifest.json"
    check_removed(NpyContainer().metadata(directory), file)
    tensors, metadata = NpyContainer().open(directory, mmap=True)
    shard_info, unsynced = _shard_info(metadata, info, file)
    try:
        info.check(shard_info, path)
        if info.version != shard_info.version:
            raise ValueError("store and shard format versions differ")
    except ValueError as error:
        raise CorruptArtifactError(str(error)) from error
    check_quantization(metadata, info, file)
    rows = metadata.get("fse.rows", "")
    if (
        metadata.get("fse.first_row") != str(first_row)
        or not INTEGER.fullmatch(rows)
        or rows == "0"
    ):
        raise CorruptArtifactError(f"{file}: the shard must hold rows from {first_row}")
    want = {"vectors", "inv_norms"} | ({"scales"} if info.dtype == "int8" else set())
    if info.has_doc_ids:
        want.add("doc_ids")
    if set(tensors) != want:
        raise CorruptArtifactError(f"{file}: a {info.dtype} shard holds exactly {sorted(want)}")
    n = int(rows)
    if tensors["vectors"].shape != (n, info.dim) or any(
        tensors[k].shape != (n,) for k in want - {"vectors"}
    ):
        raise CorruptArtifactError(f"{file}: the tensors do not hold {n} rows of {info.dim} values")
    if (
        tensors["vectors"].dtype != np.dtype(_codes.STORAGE_TYPES[info.dtype])
        or any(tensors[k].dtype != np.float32 for k in want - {"vectors", "doc_ids"})
        or (info.has_doc_ids and tensors["doc_ids"].dtype != np.dtype("<i8"))
    ):
        raise CorruptArtifactError(
            f"{file}: doc_ids must be int64; the vectors must be {info.dtype}, the inverse norms and scales float32"
        )
    expected = {key: metadata[CHECKSUM_PREFIX + key] for key in tensors}
    return Shard(
        name,
        first_row,
        n,
        unsynced,
        expected,
        parsed_stats(metadata, file) if info.has_doc_ids else None,
    ), tensors


def _shard_info(metadata: dict, info: StoreInfo, file: Path) -> tuple[StoreInfo, bool]:
    """Check the optional durability tag alongside the exact shard metadata keys."""
    content = {
        key: value
        for key, value in metadata.items()
        if key != FINGERPRINT_KEY and not key.startswith(CHECKSUM_PREFIX)
    }
    extra = _shard_keys(info)
    if info.has_doc_ids:
        extra = (*extra, *ID_KEYS)
    unsynced = "fse.durability" in content
    if unsynced:
        if content["fse.durability"] != "none":
            raise UnsupportedArtifactError(
                f"{file}: fse.durability is {content['fse.durability']!r}; "
                "this fse supports only 'none' or an absent durability tag"
            )
        extra = (*extra, "fse.durability")
    return _info(content, file, extra), unsynced


def _shard_keys(info: StoreInfo) -> tuple[str, ...]:
    """The keys a shard adds to the shared metadata."""
    keys = (CREATED_BY_KEY, "fse.first_row", "fse.rows")
    return keys if info.dtype == "float32" else (*keys, "fse.quantization")


def check_quantization(metadata: dict, info: StoreInfo, file: Path) -> None:
    """A float16 or int8 shard names the storage rule of its codes: quantize_rows@1."""
    rule = metadata.get("fse.quantization")
    if (info.dtype == "float32" and rule is None) or rule == QUANTIZATION:
        return
    try:
        parse_id(rule)
    except ValueError as error:
        raise CorruptArtifactError(f"{file}: fse.quantization must be {QUANTIZATION}") from error
    raise UnsupportedArtifactError(f"{file} is stored by {rule}; this fse reads {QUANTIZATION}")


def _blocks(array: ndarray):
    flat = array.reshape(-1)
    step = max(1, _VALUE_BYTES // array.dtype.itemsize)
    for begin in range(0, len(flat), step):
        yield flat[begin : begin + step]


def _nonnegative(shard: Shard, name: str, values: ndarray) -> None:
    for part in _blocks(values):
        if not (np.isfinite(part) & (part >= 0)).all():
            raise CorruptArtifactError(
                f"{shard.name}: {name} holds values that are not finite and at least 0"
            )


def _vectors(shard: Shard, vectors: ndarray) -> None:
    for part in _blocks(vectors):
        if vectors.dtype == np.int8:
            if (part == -128).any():
                raise CorruptArtifactError(f"{shard.name}: an int8 code is -128, beyond +-127")
        elif not np.isfinite(part).all():
            raise CorruptArtifactError(f"{shard.name}: the vectors hold values that are not finite")


class VectorKind:
    """``sentence_vectors`` as a kind of the shard store."""

    name = NAME

    def parse_root(self, version: int, metadata: Any, file: Path) -> StoreInfo:
        info = _info(metadata, file, ())
        if info.version != version:
            raise CorruptArtifactError(f"{file}: store and sentence_vectors versions differ")
        return info

    def open_shard(
        self, path: Path, info: StoreInfo, first_row: int
    ) -> tuple[Shard, dict[str, ndarray]]:
        return open_shard(path, info, first_row)

    def check_values(self, shard: Shard, tensors: dict[str, ndarray]) -> None:
        """Check values in bounded blocks, without shard-sized temporary arrays."""
        for name in ("inv_norms", "scales"):
            values = tensors.get(name)
            if values is not None:
                _nonnegative(shard, name, values)
        _vectors(shard, tensors["vectors"])
        if shard.id_stats is not None:
            verify_stats(tensors["doc_ids"], shard.id_stats, shard.name)

    def check_identifiers(self, metadata: dict, file: Path, info: StoreInfo) -> None:
        check_mode(metadata, file)
        version = metadata.get("fse.format_version")
        if isinstance(version, str) and version not in ("1", "2"):
            raise UnsupportedArtifactError(f"{file}: unsupported format {version!r}")
        durability = metadata.get("fse.durability")
        if isinstance(durability, str) and durability != "none":
            raise UnsupportedArtifactError(f"{file}: unsupported fse.durability {durability!r}")
        if isinstance(metadata.get("fse.quantization"), str):
            check_quantization(metadata, info, file)

    def summary(self, shard: Shard, tensors: dict[str, ndarray]) -> object:
        del shard, tensors  # nothing of one shard constrains its neighbours
        return None

    def follows(self, previous: object, current: object) -> str | None:
        del previous, current
        return None


VECTORS = VectorKind()
