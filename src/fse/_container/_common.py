#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The format primitives every container shares (ARCHITECTURE.md, "Artifacts").

Dtype codes, names and limits, the integrity keys, the fingerprint and checksums, strict
JSON, the checks of a write and the safe opening of regular files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy import ndarray

#: The dtype codes of the format and the numpy type descriptions they stand for.
DTYPE_CODES = {"F32": "<f4", "F16": "<f2", "I8": "|i1", "U8": "|u1", "U32": "<u4", "I64": "<i8"}


TENSOR_NAME = re.compile(r"[a-z0-9_]+(\.[a-z0-9_]+)*")


# Names safetensors reserves in its header; never names of tensors.
RESERVED_NAMES = frozenset({"__metadata__"})


# The largest dimension and number of bytes of a tensor, so numpy can describe it.
MAX_INT64 = 2**63 - 1


INTEGER = re.compile(r"0|[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")
CHECKSUM_PREFIX = "fse.checksum."
FINGERPRINT_KEY = "fse.fingerprint"
CREATED_BY_KEY = "fse.created_by"
ALIGNMENT = 64
_CHUNK = 64 * 2**20
MAX_RANK = 32
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class CorruptArtifactError(ValueError):
    """An artifact on disk breaks the format: damaged, truncated or edited."""


class TensorContainer(Protocol):
    """Writes and opens named tensors with metadata; see ARCHITECTURE.md, "Artifacts"."""

    name: str
    version: int
    suffix: str  # of the path of an artifact: "" for a directory, ".safetensors" for a file

    def write(self, path: Path, tensors: Mapping[str, ndarray], metadata: Mapping[str, str]):
        """Write a new artifact at ``path``; adds checksums and the fingerprint."""

    def open(self, path: Path, *, mmap: bool) -> tuple[dict[str, ndarray], dict[str, str]]:
        """Read-only tensors and the metadata; checks checksums unless ``mmap``."""

    def verify(self, path: Path):
        """Check every checksum against the data on disk; raises CorruptArtifactError."""


def canonical_json(value: Any) -> str:
    """JSON with sorted keys, no whitespace and non-ASCII characters unescaped."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def tensor_code(array: ndarray) -> str:
    """The dtype code of ``array``, or a ValueError for a type the format does not have.

    The format is little-endian; a big-endian array is rejected, not converted.
    """
    for code, descr in DTYPE_CODES.items():
        if array.dtype == np.dtype(descr):
            return code
    raise ValueError(
        f"dtype {array.dtype.str} is not one of the format: "
        f"{', '.join(f'{c} ({d})' for c, d in DTYPE_CODES.items())}"
    )


def fingerprint(
    entries: Mapping[str, tuple[str, tuple[int, ...], str]], metadata: Mapping[str, str]
) -> str:
    """The fingerprint of tensors given as ``name -> (code, shape, sha256)`` and metadata.

    The metadata without ``fse.created_by``, ``fse.fingerprint`` and the checksum keys;
    the tensors sorted by name (ARCHITECTURE.md, "Identities").
    """
    content = {
        k: v
        for k, v in metadata.items()
        if k not in (CREATED_BY_KEY, FINGERPRINT_KEY) and not k.startswith(CHECKSUM_PREFIX)
    }
    tensors = [
        [name, code, list(shape), digest] for name, (code, shape, digest) in sorted(entries.items())
    ]
    document = {"fingerprint": 1, "metadata": content, "tensors": tensors}
    return "sha256:" + hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def array_sha256(array: ndarray) -> str:
    """The sha256 of the little-endian C-order bytes of ``array``, in chunks."""
    digest = hashlib.sha256()
    if array.flags.c_contiguous:
        # Hash the memory itself, without a copy of the data.
        view = memoryview(array.reshape(-1).view(np.uint8))
        for begin in range(0, len(view), _CHUNK):
            digest.update(view[begin : begin + _CHUNK])
        return digest.hexdigest()
    flat = array.reshape(-1)
    step = max(1, _CHUNK // max(1, array.itemsize))
    for begin in range(0, len(flat), step):
        digest.update(flat[begin : begin + step].tobytes())
    return digest.hexdigest()


def _check_metadata(metadata: Mapping[str, str]):
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"metadata must map str to str, not {key!r}: {value!r}")


def _check_for_writing(tensors: Mapping[str, ndarray], metadata: Mapping[str, str]) -> dict:
    """Check tensors and metadata a container is asked to write; return the dtype codes."""
    _check_metadata(metadata)
    for key in metadata:
        if key == FINGERPRINT_KEY or key.startswith(CHECKSUM_PREFIX):
            raise ValueError(f"{key} is written by the container, not by the caller")
    return {name: _check_tensor(name, array) for name, array in tensors.items()}


def _check_tensor(name: Any, array: Any) -> str:
    """Check the name and array of one tensor to write; return its dtype code."""
    if not isinstance(name, str) or not TENSOR_NAME.fullmatch(name) or name in RESERVED_NAMES:
        raise ValueError(
            f"{name!r} is not a tensor name: [a-z0-9_]+ parts joined by dots, "
            f"not {', '.join(sorted(RESERVED_NAMES))}"
        )
    if not isinstance(array, ndarray):
        raise TypeError(f"tensor {name} must be a numpy array, not {type(array).__name__}")
    if not array.flags.c_contiguous:
        raise ValueError(f"tensor {name} must be C-contiguous")
    if array.ndim > MAX_RANK:
        raise ValueError(f"tensor {name} has more than {MAX_RANK} dimensions")
    code = tensor_code(array)
    if not _shape(list(array.shape), array.itemsize):
        raise ValueError(f"the shape of tensor {name} is beyond what readers accept")
    return code


def _integrity(
    path: Path, specs: Mapping[str, tuple[str, tuple[int, ...]]], metadata: dict
) -> dict:
    """Check the checksum keys and the fingerprint of tensors given as ``name -> (code,
    shape)``; return ``name -> (code, shape, checksum)``.
    """
    checksums = {
        k[len(CHECKSUM_PREFIX) :]: v for k, v in metadata.items() if k.startswith(CHECKSUM_PREFIX)
    }
    if set(checksums) != set(specs):
        raise CorruptArtifactError(f"{path}: the checksum keys do not match the tensors")
    for name, digest in checksums.items():
        if not _SHA256.fullmatch(digest):
            raise CorruptArtifactError(f"{path}: the checksum of {name} is malformed")
    recorded = metadata.get(FINGERPRINT_KEY)
    if recorded is None or not _FINGERPRINT.fullmatch(recorded):
        raise CorruptArtifactError(f"{path}: {FINGERPRINT_KEY} is missing or malformed")
    entries = {name: (code, shape, checksums[name]) for name, (code, shape) in specs.items()}
    if fingerprint(entries, metadata) != recorded:
        raise CorruptArtifactError(f"{path}: {FINGERPRINT_KEY} does not match the content")
    return entries


def _strict_json(raw: bytes, file: Path) -> Any:
    """Parse JSON without duplicate keys and without NaN or infinity."""

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise CorruptArtifactError(f"{file} has a duplicate key")
        return dict(items)

    def constant(name):
        raise CorruptArtifactError(f"{file} contains {name}, which is not JSON")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except CorruptArtifactError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as err:
        # ValueError covers json.JSONDecodeError and integers beyond Python's digit limit,
        # RecursionError a nesting too deep for the parser.
        raise CorruptArtifactError(f"{file} is not valid JSON: {err}") from err
    _check_strings(value, file)
    return value


def _strings(value: Any):
    """Every string of a parsed JSON value, keys included, without recursion."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _check_strings(value: Any, file: Path):
    """Reject strings without a UTF-8 form (lone surrogates from \\u escapes)."""
    for text in _strings(value):
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise CorruptArtifactError(f"{file} holds a string that is not valid UTF-8") from None


def _shape(value: Any, itemsize: int) -> bool:
    """A shape numpy can describe: at most MAX_RANK non-negative int64 dimensions whose
    non-zero product, in bytes, stays within int64 (also for tensors with a zero dimension)."""
    if not isinstance(value, list) or len(value) > MAX_RANK:
        return False
    if not all(
        isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= MAX_INT64 for n in value
    ):
        return False
    return math.prod(n for n in value if n) * itemsize <= MAX_INT64


def _open_regular(file: Path):
    """Open ``file`` for reading if it is a regular file and no symbolic link.

    The descriptor is compared with the checked path, so a file swapped in between is
    caught; everything is read and mapped through this one handle.
    """
    try:
        before = os.lstat(file)
    except FileNotFoundError:
        raise CorruptArtifactError(f"{file} is missing") from None
    if not stat.S_ISREG(before.st_mode):
        raise CorruptArtifactError(f"{file} is not a regular file (a symbolic link or else)")
    # Without following a link, and without blocking if a FIFO was swapped in meanwhile.
    extra = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        handle = open(file, "rb", opener=lambda name, flags: os.open(name, flags | extra))  # noqa: SIM115
    except OSError as err:
        raise CorruptArtifactError(f"{file} cannot be opened as a regular file: {err}") from err
    after = os.fstat(handle.fileno())
    if not stat.S_ISREG(after.st_mode) or (after.st_dev, after.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        handle.close()
        raise CorruptArtifactError(f"{file} changed while it was opened")
    return handle


def _sync_directory(path: Path):
    """Make the entries of a directory durable (POSIX; Windows has no directory fsync)."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
