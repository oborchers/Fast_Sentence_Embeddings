#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""NpyContainer 1: a directory of .npy files with a manifest, for stored sentence vectors."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import secrets
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray
from numpy.lib import format as npy_format

from fse._contract import UnsupportedArtifactError

from ._common import (
    _CHUNK,
    _EMPTY_SHA256,
    ALIGNMENT,
    CHECKSUM_PREFIX,
    DTYPE_CODES,
    FINGERPRINT_KEY,
    RESERVED_NAMES,
    TENSOR_NAME,
    CorruptArtifactError,
    _check_for_writing,
    _integrity,
    _open_regular,
    _shape,
    _strict_json,
    _sync_directory,
    fingerprint,
    tensor_code,
)

MANIFEST = "manifest.json"


# Limits that keep a damaged or crafted artifact from making the reader allocate much
# before anything is checked: the manifest, a .npy header, and the rank of a tensor.
MAX_MANIFEST_BYTES = 16 * 2**20


MAX_HEADER_BYTES = 65_536


def _read_manifest(path: Path) -> dict[str, Any]:
    file = path / MANIFEST
    raw = _manifest_bytes(path, file)
    manifest = _strict_json(raw, file)
    if not isinstance(manifest, dict) or set(manifest) != {
        "container",
        "container_version",
        "tensors",
        "metadata",
    }:
        raise CorruptArtifactError(
            f"{file} must hold exactly container, container_version, tensors and metadata"
        )
    _check_version(path, manifest)
    if not isinstance(manifest["tensors"], dict) or not isinstance(manifest["metadata"], dict):
        raise CorruptArtifactError(f"{file}: tensors and metadata must be objects")
    for key, value in manifest["metadata"].items():
        if not isinstance(value, str):
            raise CorruptArtifactError(f"{file}: metadata {key} must be a string")
    for name, entry in manifest["tensors"].items():
        _check_manifest_tensor(file, name, entry)
    return manifest


def _manifest_bytes(path: Path, file: Path) -> bytes:
    """The bytes of the manifest ``file`` of the artifact directory ``path``, bounded."""
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory with an artifact")
    if not os.path.lexists(file):
        raise CorruptArtifactError(f"{path} has no {MANIFEST}")
    with _open_regular(file) as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > MAX_MANIFEST_BYTES:
            raise CorruptArtifactError(f"{file} is larger than {MAX_MANIFEST_BYTES} bytes")
        return handle.read(size)


def _check_version(path: Path, manifest: dict[str, Any]):
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


def _check_manifest_tensor(file: Path, name: str, entry: Any):
    if not TENSOR_NAME.fullmatch(name) or name in RESERVED_NAMES:
        raise CorruptArtifactError(f"{file}: {name!r} is not a tensor name")
    if not isinstance(entry, dict) or set(entry) != {"file", "dtype", "shape"}:
        raise CorruptArtifactError(f"{file}: tensor {name} must hold exactly file, dtype and shape")
    if entry["file"] != name + ".npy":
        raise CorruptArtifactError(f"{file}: tensor {name} must be stored in {name}.npy")
    if not isinstance(entry["dtype"], str) or entry["dtype"] not in DTYPE_CODES:
        raise CorruptArtifactError(
            f"{file}: tensor {name} has the unknown dtype {entry['dtype']!r}"
        )
    shape = entry["shape"]
    if not _shape(shape, np.dtype(DTYPE_CODES[entry["dtype"]]).itemsize):
        raise CorruptArtifactError(
            f"{file}: the shape of tensor {name} must be non-negative ints that numpy can hold"
        )


def _declared_header(handle, file: Path, size: int) -> tuple[int, int]:
    """The width of the length field and the declared header length of a .npy file,
    checked before numpy reads that many bytes."""
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
    return width, declared


def _read_header(handle, file: Path, code: str, shape: tuple[int, ...]) -> int:
    """Check the .npy header of ``handle`` and return the offset of its data."""
    size = os.fstat(handle.fileno()).st_size
    width, declared = _declared_header(handle, file, size)
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


def write_manifest(
    path: Path,
    entries: Mapping[str, tuple[str, tuple[int, ...], str]],
    metadata: Mapping[str, str],
    tmp: Path,
    sync=os.fsync,
):
    """Commit the artifact in directory ``path`` whose tensor files are written and
    synced: ``entries`` holds (dtype code, shape, sha256) per tensor. The manifest (with
    the checksums and the fingerprint) is written to ``tmp`` in ``path``, synced by
    ``sync`` and renamed into place last; without it, readers see no artifact."""
    full = dict(metadata)
    for name, (_, _, digest) in entries.items():
        full[CHECKSUM_PREFIX + name] = digest
    full[FINGERPRINT_KEY] = fingerprint(dict(entries), full)
    manifest = {
        "container": NpyContainer.name,
        "container_version": NpyContainer.version,
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
        sync(f.fileno())
    _sync_directory(path)
    os.replace(tmp, path / MANIFEST)
    _sync_directory(path)


class NpyContainer:
    """One ``<name>.npy`` file per tensor and a ``manifest.json``, in a directory."""

    name = "npy"
    version = 1
    suffix = ""  # a directory

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
        tensors = {
            n: a.view() if isinstance(a, ndarray) else a for n, a in tensors.items()
        }  # snapshots
        metadata = dict(metadata)
        codes = _check_for_writing(tensors, metadata)
        # Creating the directory reserves the path atomically; the manifest, renamed into
        # place last, commits the artifact: without it, readers see no artifact.
        path.mkdir()
        tmp = path / f".{MANIFEST}.writing-{secrets.token_hex(4)}"
        try:
            entries = {}
            for name, array in tensors.items():
                digest = self._write_tensor(path / f"{name}.npy", array)
                if expected is not None and expected.get(name) != digest:
                    raise CorruptArtifactError(
                        f"tensor {name} changed since it was checked; nothing was written"
                    )
                entries[name] = (codes[name], tuple(int(n) for n in array.shape), digest)
            write_manifest(path, entries, metadata, tmp)
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
        specs = {n: (e["dtype"], tuple(e["shape"])) for n, e in manifest["tensors"].items()}
        return _integrity(path, specs, metadata)

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
