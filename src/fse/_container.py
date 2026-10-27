#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Artifacts as named tensors with string metadata, and their first container.

ARCHITECTURE.md, "Artifacts" and "NpyContainer 1", specifies the format. A container
writes a mapping of names to numpy arrays plus a flat ``str`` to ``str`` metadata map and
opens it again, bit for bit. It adds the integrity keys itself: the sha256 of every
tensor (``fse.checksum.<name>``) and the fingerprint of the content (``fse.fingerprint``),
which does not depend on the container.

Only the standard library and numpy are used, so the 1.x export can copy this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy import ndarray
from numpy.lib import format as npy_format

from fse._contract import UnsupportedArtifactError

#: The dtype codes of the format and the numpy type descriptions they stand for.
DTYPE_CODES = {"F32": "<f4", "F16": "<f2", "I8": "|i1", "U8": "|u1", "U32": "<u4", "I64": "<i8"}

TENSOR_NAME = re.compile(r"[a-z0-9_]+(\.[a-z0-9_]+)*")
INTEGER = re.compile(r"0|[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")

CHECKSUM_PREFIX = "fse.checksum."
FINGERPRINT_KEY = "fse.fingerprint"
CREATED_BY_KEY = "fse.created_by"

ALIGNMENT = 64
_CHUNK = 64 * 2**20
_MANIFEST = "manifest.json"
# Limits that keep a damaged or crafted artifact from making the reader allocate much
# before anything is checked: the manifest, a .npy header, and the rank of a tensor.
MAX_MANIFEST_BYTES = 16 * 2**20
MAX_HEADER_BYTES = 65_536
MAX_RANK = 32
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class CorruptArtifactError(ValueError):
    """An artifact on disk breaks the format: damaged, truncated or edited."""


class TensorContainer(Protocol):
    """Writes and opens named tensors with metadata; see ARCHITECTURE.md, "Artifacts"."""

    name: str
    version: int

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


def _read_manifest(path: Path) -> dict[str, Any]:
    file = path / _MANIFEST
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory with an artifact")
    if not os.path.lexists(file):
        raise CorruptArtifactError(f"{path} has no {_MANIFEST}")
    with _open_regular(file) as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > MAX_MANIFEST_BYTES:
            raise CorruptArtifactError(f"{file} is larger than {MAX_MANIFEST_BYTES} bytes")
        raw = handle.read(size)

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise CorruptArtifactError(f"{file} has a duplicate key")
        return dict(items)

    def constant(name):
        raise CorruptArtifactError(f"{file} contains {name}, which is not JSON")

    try:
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise CorruptArtifactError(f"{file} is not valid JSON: {err}") from err
    if not isinstance(manifest, dict) or set(manifest) != {
        "container",
        "container_version",
        "tensors",
        "metadata",
    }:
        raise CorruptArtifactError(
            f"{file} must hold exactly container, container_version, tensors and metadata"
        )
    version = manifest["container_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or manifest["container"] != NpyContainer.name
        or manifest["container_version"] != NpyContainer.version
    ):
        raise UnsupportedArtifactError(
            f"{path} is written by container {manifest['container']!r} version "
            f"{manifest['container_version']!r}; this fse reads {NpyContainer.name!r} "
            f"version {NpyContainer.version}"
        )
    if not isinstance(manifest["tensors"], dict) or not isinstance(manifest["metadata"], dict):
        raise CorruptArtifactError(f"{file}: tensors and metadata must be objects")
    for key, value in manifest["metadata"].items():
        if not isinstance(value, str):
            raise CorruptArtifactError(f"{file}: metadata {key} must be a string")
    for name, entry in manifest["tensors"].items():
        if not TENSOR_NAME.fullmatch(name):
            raise CorruptArtifactError(f"{file}: {name!r} is not a tensor name")
        if not isinstance(entry, dict) or set(entry) != {"file", "dtype", "shape"}:
            raise CorruptArtifactError(
                f"{file}: tensor {name} must hold exactly file, dtype and shape"
            )
        if entry["file"] != name + ".npy":
            raise CorruptArtifactError(f"{file}: tensor {name} must be stored in {name}.npy")
        if not isinstance(entry["dtype"], str) or entry["dtype"] not in DTYPE_CODES:
            raise CorruptArtifactError(
                f"{file}: tensor {name} has the unknown dtype {entry['dtype']!r}"
            )
        shape = entry["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) > MAX_RANK
            or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape)
        ):
            raise CorruptArtifactError(
                f"{file}: the shape of tensor {name} must be non-negative ints"
            )
    return manifest


def _read_header(handle, file: Path, code: str, shape: tuple[int, ...]) -> int:
    """Check the .npy header of ``handle`` and return the offset of its data."""
    size = os.fstat(handle.fileno()).st_size
    # The declared header length, checked before numpy reads that many bytes.
    prefix = handle.read(12)
    if len(prefix) < 10 or prefix[:6] != b"\x93NUMPY":
        raise CorruptArtifactError(f"{file} has no valid .npy header: no .npy magic")
    if prefix[6:8] not in (b"\x01\x00", b"\x02\x00"):
        raise CorruptArtifactError(
            f"{file} has the unsupported .npy version {(prefix[6], prefix[7])}"
        )
    width = 2 if prefix[6] == 1 else 4
    declared = int.from_bytes(prefix[8 : 8 + width], "little")
    if declared > MAX_HEADER_BYTES or 8 + width + declared > size:
        raise CorruptArtifactError(f"{file} declares a header of {declared} bytes")
    # numpy parses one bounded copy of the header, never the file itself again.
    handle.seek(0)
    raw = handle.read(8 + width + declared)
    if len(raw) != 8 + width + declared:
        raise CorruptArtifactError(f"{file} ended inside its header")
    buffer = io.BytesIO(raw)
    try:
        npy_format.read_magic(buffer)
        reader = (
            npy_format.read_array_header_1_0 if width == 2 else npy_format.read_array_header_2_0
        )
        header = reader(buffer, max_header_size=MAX_HEADER_BYTES)
    except (ValueError, OSError) as err:
        raise CorruptArtifactError(f"{file} has no valid .npy header: {err}") from err
    found_shape, fortran, dtype = header
    if dtype.hasobject or dtype.str != DTYPE_CODES[code]:
        raise CorruptArtifactError(
            f"{file} holds {dtype.str}, the manifest says {code} ({DTYPE_CODES[code]})"
        )
    if fortran:
        raise CorruptArtifactError(f"{file} is in Fortran order")
    if tuple(found_shape) != shape:
        raise CorruptArtifactError(f"{file} has shape {found_shape}, the manifest says {shape}")
    offset = len(raw)
    if offset % ALIGNMENT:
        raise CorruptArtifactError(
            f"the data of {file} starts at {offset}, not at a multiple of {ALIGNMENT}"
        )
    nbytes = math.prod(shape) * dtype.itemsize  # Python ints: no overflow
    if size != offset + nbytes:
        raise CorruptArtifactError(f"{file} has {size} bytes, {offset + nbytes} expected")
    return offset


def _sync_directory(path: Path):
    """Make the entries of a directory durable (POSIX; Windows has no directory fsync)."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class NpyContainer:
    """One ``<name>.npy`` file per tensor and a ``manifest.json``, in a directory."""

    name = "npy"
    version = 1

    def write(
        self,
        path,
        tensors: Mapping[str, ndarray],
        metadata: Mapping[str, str],
        expected: Mapping[str, str] | None = None,
    ):
        """Write a new artifact; ``expected`` checksums, if given, must match the data.

        With ``expected``, every tensor is hashed while it is written and compared with
        its expected checksum, so data that changed since it was checked (a mapped file
        changed by another process) is never committed.
        """
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{path} exists; an artifact is never overwritten")
        _check_metadata(metadata)
        for key in metadata:
            if key == FINGERPRINT_KEY or key.startswith(CHECKSUM_PREFIX):
                raise ValueError(f"{key} is written by the container, not by the caller")
        codes = {}
        for name, array in tensors.items():
            if not isinstance(name, str) or not TENSOR_NAME.fullmatch(name):
                raise ValueError(f"{name!r} is not a tensor name: [a-z0-9_]+ parts joined by dots")
            if not isinstance(array, ndarray):
                raise TypeError(f"tensor {name} must be a numpy array, not {type(array).__name__}")
            if not array.flags.c_contiguous:
                raise ValueError(f"tensor {name} must be C-contiguous")
            if array.ndim > MAX_RANK:
                raise ValueError(f"tensor {name} has more than {MAX_RANK} dimensions")
            codes[name] = tensor_code(array)
        # Creating the directory reserves the path atomically; the manifest, renamed into
        # place last, commits the artifact: without it, readers see no artifact.
        path.mkdir()
        tmp = path / f".{_MANIFEST}.writing-{secrets.token_hex(4)}"
        try:
            entries = {}
            for name, array in tensors.items():
                digest = self._write_tensor(path / f"{name}.npy", array)
                if expected is not None and expected.get(name) != digest:
                    raise CorruptArtifactError(
                        f"tensor {name} changed since it was checked; nothing was written"
                    )
                entries[name] = (codes[name], tuple(int(n) for n in array.shape), digest)
            full = dict(metadata)
            for name, (_, _, digest) in entries.items():
                full[CHECKSUM_PREFIX + name] = digest
            full[FINGERPRINT_KEY] = fingerprint(entries, full)
            manifest = {
                "container": self.name,
                "container_version": self.version,
                "tensors": {
                    name: {"file": f"{name}.npy", "dtype": code, "shape": list(shape)}
                    for name, (code, shape, _) in entries.items()
                },
                "metadata": full,
            }
            text = json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
            if len(text.encode("utf-8")) > MAX_MANIFEST_BYTES:
                raise ValueError(f"the manifest would exceed {MAX_MANIFEST_BYTES} bytes")
            with tmp.open("w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            _sync_directory(path)
            os.replace(tmp, path / _MANIFEST)
            _sync_directory(path)
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        _sync_directory(path.parent)

    @staticmethod
    def _write_tensor(file: Path, array: ndarray) -> str:
        digest = hashlib.sha256()
        data = array.astype(DTYPE_CODES[tensor_code(array)], copy=False)
        with file.open("wb") as f:
            npy_format.write_array_header_1_0(f, npy_format.header_data_from_array_1_0(data))
            flat = data.reshape(-1)
            step = max(1, _CHUNK // max(1, data.itemsize))
            for begin in range(0, len(flat), step):
                chunk = flat[begin : begin + step].tobytes()
                digest.update(chunk)
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        return digest.hexdigest()

    def open(self, path, *, mmap: bool) -> tuple[dict[str, ndarray], dict[str, str]]:
        path = Path(path)
        manifest = _read_manifest(path)
        metadata: dict[str, str] = dict(manifest["metadata"])
        entries = self._check_integrity_keys(path, manifest, metadata)
        tensors = {}
        for name, entry in manifest["tensors"].items():
            file = path / entry["file"]
            shape = tuple(entry["shape"])
            code = entry["dtype"]
            with _open_regular(file) as handle:
                offset = _read_header(handle, file, code, shape)
                if 0 in shape:
                    if entries[name][2] != _EMPTY_SHA256:
                        raise CorruptArtifactError(f"{file} does not match its checksum")
                    array = np.empty(shape, dtype=DTYPE_CODES[code])  # nothing to map
                    array.setflags(write=False)
                elif mmap:
                    array = np.memmap(
                        handle, dtype=DTYPE_CODES[code], mode="r", offset=offset, shape=shape
                    )
                else:
                    array = self._read(handle, file, code, shape, entries[name][2])
            tensors[name] = array
        return tensors, metadata

    @staticmethod
    def _read(handle, file: Path, code: str, shape: tuple[int, ...], expected: str) -> ndarray:
        """Read a tensor into immutable ``bytes``, checked against its checksum.

        An array over ``bytes`` cannot be made writable again, so the data cannot change.
        """
        nbytes = math.prod(shape) * np.dtype(DTYPE_CODES[code]).itemsize
        data = handle.read(nbytes)
        if len(data) != nbytes:
            raise CorruptArtifactError(f"{file} ended early")
        digest = hashlib.sha256()
        view = memoryview(data)
        for begin in range(0, nbytes, _CHUNK):
            digest.update(view[begin : begin + _CHUNK])
        if digest.hexdigest() != expected:
            raise CorruptArtifactError(f"{file} does not match its checksum")
        return np.frombuffer(data, dtype=DTYPE_CODES[code]).reshape(shape)

    @staticmethod
    def _check_integrity_keys(path: Path, manifest: dict, metadata: dict[str, str]) -> dict:
        tensors = manifest["tensors"]
        checksums = {
            k[len(CHECKSUM_PREFIX) :]: v
            for k, v in metadata.items()
            if k.startswith(CHECKSUM_PREFIX)
        }
        if set(checksums) != set(tensors):
            raise CorruptArtifactError(f"{path}: the checksum keys do not match the tensors")
        for name, digest in checksums.items():
            if not _SHA256.fullmatch(digest):
                raise CorruptArtifactError(f"{path}: the checksum of {name} is malformed")
        recorded = metadata.get(FINGERPRINT_KEY)
        if recorded is None or not _FINGERPRINT.fullmatch(recorded):
            raise CorruptArtifactError(f"{path}: {FINGERPRINT_KEY} is missing or malformed")
        entries = {
            name: (entry["dtype"], tuple(entry["shape"]), checksums[name])
            for name, entry in tensors.items()
        }
        if fingerprint(entries, metadata) != recorded:
            raise CorruptArtifactError(f"{path}: {FINGERPRINT_KEY} does not match the content")
        return entries

    def verify(self, path):
        path = Path(path)
        manifest = _read_manifest(path)
        metadata = dict(manifest["metadata"])
        entries = self._check_integrity_keys(path, manifest, metadata)
        for name, entry in manifest["tensors"].items():
            file = path / entry["file"]
            with _open_regular(file) as handle:
                _read_header(handle, file, entry["dtype"], tuple(entry["shape"]))
                digest = hashlib.sha256()
                while chunk := handle.read(_CHUNK):
                    digest.update(chunk)
            if digest.hexdigest() != entries[name][2]:
                raise CorruptArtifactError(f"{file} does not match its checksum")


#: Every container this fse can write and read; the golden test runs over all of them.
CONTAINERS: dict[str, TensorContainer] = {NpyContainer.name: NpyContainer()}
