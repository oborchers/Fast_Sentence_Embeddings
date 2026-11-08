#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The companion artifact of the sign bits of one shard (docs/architecture/store.md, "Sign bits").

Next to a shard ``shard-<first row>`` of a store of sentence vectors may lie the artifact
``sign_bits-<first row>``: an NpyContainer with the tensor ``sign_bits`` (uint8, one row of
``ceil(dim / 8)`` bytes per row of the shard) made by the rule ``sign_bits@1`` of
``fse._reference_bits``. The shard, ``store.json`` and their fingerprints are unchanged by it;
it is bound to its shard by ``fse.shard_fingerprint``. A shard without the artifact is valid:
the bits can be derived from its vectors at any time. An artifact that is not the valid one of
a shard in the snapshot is damage, and one of an unknown version or rule is never skipped.

This module knows the names and the metadata of the artifact, opens and checks it. Writing it is
``_bits_write``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _reference_bits
from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    CorruptArtifactError,
    NpyContainer,
    checksum,
)
from fse._container._common import _check_algorithm, _strict_json
from fse._container._npy import _check_version, _manifest_bytes
from fse._contract import UnsupportedArtifactError

from ._kind import Shard, shard_name
from ._vectors_format import NAME as VECTORS_NAME

PREFIX = "sign_bits-"
TENSOR = "sign_bits"
KIND = "sign_bits"
FORMAT_VERSION = "1"
RULE = "sign_bits@1"
DURABILITY = "fse.durability"
_BLOCK_BYTES = 1 << 20  # the bool matrix of the recomputation of a check
_KEYS = (
    "fse.kind",
    "fse.format_version",
    "fse.bits",
    "fse.encoder",
    "fse.dtype",
    "fse.output_dim",
    "fse.first_row",
    "fse.rows",
    "fse.shard_fingerprint",
    CREATED_BY_KEY,
)


def bits_name(first_row: int) -> str:
    return f"{PREFIX}{first_row:016d}"


def width(dim: int) -> int:
    """The bytes of the bits of one row."""
    return -(-dim // 8)


def has_bits(info: Any) -> bool:
    """Whether the store of ``info`` is of a kind that has sign bits (sentence vectors)."""
    return info.kind == VECTORS_NAME


def present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def canonical_bits(path: Path) -> list[tuple[int, str]]:
    """The canonical entries ``sign_bits-<first row>`` of the store; dot entries and other
    names are ignored."""
    found = []
    for entry in path.iterdir():
        suffix = entry.name.removeprefix(PREFIX)
        if entry.name.startswith(PREFIX) and suffix.isdecimal():
            row = int(suffix)
            if entry.name == bits_name(row):
                found.append((row, entry.name))
    return sorted(found)


def first_orphan(path: Path, rows: set[int], end: int) -> tuple[int, str] | None:
    """The first artifact of no shard of the snapshot (the first rows of its shards are
    ``rows``, it ends at row ``end``), or None. One at or after ``end`` whose shard is
    published now belongs to a writer that published the shard after the snapshot, and is passed
    over: a writer publishes the bits only after the shard."""
    for row, name in canonical_bits(path):
        if row in rows:
            continue
        if row < end or not present(path / shard_name(row)):
            return row, name
    return None


def metadata_of(
    info: Any, shard: Shard, created_by: str, *, unsynced: bool = False
) -> dict[str, str]:
    """The metadata of the artifact of ``shard`` (``fse.durability=none`` if ``unsynced``)."""
    metadata = {
        "fse.kind": KIND,
        "fse.format_version": FORMAT_VERSION,
        "fse.bits": RULE,
        "fse.encoder": info.encoder,
        "fse.dtype": info.dtype,
        "fse.output_dim": str(info.dim),
        "fse.first_row": str(shard.first_row),
        "fse.rows": str(shard.rows),
        "fse.shard_fingerprint": shard.fingerprint,
        CREATED_BY_KEY: created_by,
    }
    if unsynced:
        metadata[DURABILITY] = "none"
    return metadata


def check_identifiers(metadata: dict, file: Path) -> None:
    """A readable future version or rule is an UnsupportedArtifactError, before today's
    structure rules apply."""
    version = metadata.get("fse.format_version")
    if isinstance(version, str) and version != FORMAT_VERSION:
        raise UnsupportedArtifactError(
            f"{file}: sign bits of format {version!r}; this fse reads format {FORMAT_VERSION}"
        )
    rule = metadata.get("fse.bits")
    if isinstance(rule, str) and rule != RULE:
        raise UnsupportedArtifactError(
            f"{file}: sign bits by the rule {rule!r}; this fse reads {RULE}"
        )
    durability = metadata.get(DURABILITY)
    if isinstance(durability, str) and durability != "none":
        raise UnsupportedArtifactError(f"{file}: unsupported {DURABILITY} {durability!r}")


def check_manifest(directory: Path) -> None:
    """The readable identifiers of the artifact in ``directory`` before today's structure rules
    apply: a container, container version, format version, rule, durability tag or checksum
    algorithm of the future is an UnsupportedArtifactError (damage that cannot be read is a
    CorruptArtifactError)."""
    file = directory / "manifest.json"
    manifest = _strict_json(_manifest_bytes(directory, file), file)
    if not isinstance(manifest, dict):
        return
    container = manifest.get("container")
    if isinstance(container, str) and container != NpyContainer.name:
        raise UnsupportedArtifactError(f"{file}: unsupported container {container!r}")
    if "container" in manifest and "container_version" in manifest:
        _check_version(directory, manifest)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return
    check_identifiers(metadata, file)
    values = [
        value
        for key, value in metadata.items()
        if isinstance(value, str) and (key.startswith(CHECKSUM_PREFIX) or key == FINGERPRINT_KEY)
    ]
    if values:
        _check_algorithm(directory, values)


def check_directory(path: Path, name: str) -> Path:
    directory = path / name
    if not directory.is_dir() or directory.is_symlink():
        raise CorruptArtifactError(f"{name} is not a regular directory")
    return directory


def _tagged(metadata: dict, file: Path) -> bool:
    """Whether the artifact is tagged ``fse.durability=none``, after the exact keys."""
    content = {
        key for key in metadata if key != FINGERPRINT_KEY and not key.startswith(CHECKSUM_PREFIX)
    }
    tagged = DURABILITY in content
    if content != {*_KEYS, *((DURABILITY,) if tagged else ())}:
        raise CorruptArtifactError(f"{file}: the metadata must hold exactly {', '.join(_KEYS)}")
    return tagged


def _check_bound(metadata: dict, info: Any, shard: Shard, file: Path) -> None:
    for key, want in (
        ("fse.kind", KIND),
        ("fse.encoder", info.encoder),
        ("fse.dtype", info.dtype),
        ("fse.output_dim", str(info.dim)),
        ("fse.first_row", str(shard.first_row)),
        ("fse.rows", str(shard.rows)),
    ):
        if metadata[key] != want:
            raise CorruptArtifactError(f"{file}: {key} is {metadata[key]!r}, not {want!r}")
    if metadata["fse.shard_fingerprint"] != shard.fingerprint:
        raise CorruptArtifactError(f"{file}: it is bound to another shard than {shard.name}")


def open_bits(path: Path, info: Any, shard: Shard, *, verify: bool) -> tuple[ndarray, int] | None:
    """The mapped bits of ``shard`` of the store ``path`` with the bytes hashed, or None if the
    artifact does not exist. Its structure, tensor and binding to the shard are checked; an
    artifact tagged ``fse.durability=none`` is hashed if ``verify``. A damaged artifact is a
    CorruptArtifactError, an unknown version or rule an UnsupportedArtifactError."""
    name = bits_name(shard.first_row)
    if not present(path / name):
        return None
    directory = check_directory(path, name)
    file = directory / "manifest.json"
    check_manifest(directory)
    tensors, metadata = NpyContainer().open(directory, mmap=True)
    tagged = _tagged(metadata, file)
    _check_bound(metadata, info, shard, file)
    bits = tensors.get(TENSOR)
    if (
        set(tensors) != {TENSOR}
        or bits is None
        or bits.dtype != np.uint8
        or bits.shape != (shard.rows, width(info.dim))
    ):
        raise CorruptArtifactError(
            f"{file}: it must hold exactly {TENSOR}, uint8, {shard.rows} rows of "
            f"{width(info.dim)} bytes"
        )
    hashed = 0
    if verify and tagged:
        if checksum(bits) != metadata[CHECKSUM_PREFIX + TENSOR]:
            raise CorruptArtifactError(f"{name}: the checksum of {TENSOR} does not match")
        hashed = bits.nbytes
    return bits, hashed


def attached(path: Path, info: Any, shard: Shard, *, verify: bool) -> tuple[Shard, int]:
    """``shard`` with its bits mapped if the artifact exists, and the bytes hashed for it."""
    if not has_bits(info):
        return shard, 0
    found = open_bits(path, info, shard, verify=verify)
    if found is None:
        return shard, 0
    return replace(shard, sign_bits=found[0]), found[1]


def check_content(directory: Path, shard: Shard, vectors: ndarray) -> None:
    """Hash the artifact on disk against its checksum, then check that the padding bits are 0
    and that the bits are those of the vectors of the shard, in blocks."""
    bits = shard.sign_bits
    assert bits is not None
    NpyContainer().verify(directory)
    dim = vectors.shape[1]
    pad = (8 - dim % 8) % 8
    if pad and (bits[:, -1] & ((1 << pad) - 1)).any():
        raise CorruptArtifactError(f"{directory.name}: the padding bits are not 0")
    step = max(1, _BLOCK_BYTES // dim)
    for begin in range(0, shard.rows, step):
        derived = _reference_bits.sign_bits(vectors[begin : begin + step])
        if not np.array_equal(derived, bits[begin : begin + step]):
            raise CorruptArtifactError(
                f"{directory.name}: the bits are not those of the vectors of {shard.name}"
            )
