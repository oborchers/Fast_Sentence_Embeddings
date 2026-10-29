#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""safetensors 1: one file of tensors with string metadata, for models."""

from __future__ import annotations

import io
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import UnsupportedArtifactError

from . import _hash
from ._common import (
    _CHUNK,
    ALIGNMENT,
    CHECKSUM_PREFIX,
    DTYPE_CODES,
    FINGERPRINT_KEY,
    TENSOR_NAME,
    CorruptArtifactError,
    _check_for_writing,
    _integrity,
    _open_regular,
    _shape,
    _strict_json,
    _sync_directory,
    canonical_json,
    fingerprint,
)

MAX_SAFETENSORS_HEADER_BYTES = 16 * 2**20
_SAFETENSORS_LENGTH = 8  # bytes of the little-endian header length


def _read_safetensors_header(
    handle, file: Path, size: int | None = None
) -> tuple[int, dict, dict[str, str]]:
    """Parse and check the header of a safetensors file (of ``size`` bytes, by default
    that of the open file).

    Returns the offset of the data, ``name -> (code, shape, begin, end)`` and the metadata.
    """
    size = os.fstat(handle.fileno()).st_size if size is None else size
    length, header = _header_json(handle, file, size)
    metadata = _header_metadata(file, header)
    start = _SAFETENSORS_LENGTH + length
    if start % 8:
        raise CorruptArtifactError(f"the data of {file} starts at {start}, not at a multiple of 8")
    entries = {name: _tensor_entry(file, name, entry) for name, entry in header.items()}
    _check_coverage(file, entries, start, size)
    return start, entries, metadata


def _header_json(handle, file: Path, size: int) -> tuple[int, dict]:
    """The declared length and the JSON object of the header, read within its limits."""
    prefix = handle.read(_SAFETENSORS_LENGTH)
    if len(prefix) != _SAFETENSORS_LENGTH:
        raise CorruptArtifactError(f"{file} is too short for a safetensors file")
    length = int.from_bytes(prefix, "little")
    if length > MAX_SAFETENSORS_HEADER_BYTES or _SAFETENSORS_LENGTH + length > size:
        raise CorruptArtifactError(f"{file} declares a header of {length} bytes")
    raw = handle.read(length)
    if len(raw) != length:
        raise CorruptArtifactError(f"{file} ended inside its header")
    # The JSON starts with "{" and may be padded with spaces only.
    text = raw.rstrip(b" ")
    if not text.startswith(b"{") or not text.endswith(b"}"):
        raise CorruptArtifactError(f"{file} has a header that is no JSON object padded with spaces")
    header = _strict_json(text, file)
    if not isinstance(header, dict):
        raise CorruptArtifactError(f"{file} has a header that is no JSON object")
    return length, header


def _header_metadata(file: Path, header: dict) -> dict[str, str]:
    """Take ``__metadata__`` out of the header: a mapping of strings to strings."""
    metadata = header.pop("__metadata__", None)
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise CorruptArtifactError(f"{file} needs __metadata__ mapping strings to strings")
    return metadata


def _tensor_entry(file: Path, name: str, entry: Any) -> tuple[str, tuple[int, ...], int, int]:
    """Check one tensor of the header; return its code, shape and data offsets."""
    if not TENSOR_NAME.fullmatch(name):
        raise CorruptArtifactError(f"{file}: {name!r} is not a tensor name")
    if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
        raise CorruptArtifactError(
            f"{file}: tensor {name} must hold exactly dtype, shape and data_offsets"
        )
    code = entry["dtype"]
    if not isinstance(code, str):
        raise CorruptArtifactError(f"{file}: tensor {name} has no dtype string")
    if code not in DTYPE_CODES:
        raise UnsupportedArtifactError(
            f"{file}: tensor {name} has dtype {code}; fse reads {', '.join(DTYPE_CODES)}"
        )
    if not _shape(entry["shape"], np.dtype(DTYPE_CODES[code]).itemsize):
        raise CorruptArtifactError(
            f"{file}: the shape of tensor {name} must be non-negative ints that numpy can hold"
        )
    offsets = entry["data_offsets"]
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in offsets)
    ):
        raise CorruptArtifactError(f"{file}: tensor {name} needs two data offsets")
    begin, end = offsets
    itemsize = np.dtype(DTYPE_CODES[code]).itemsize
    shape = tuple(entry["shape"])
    if end - begin != math.prod(shape) * itemsize:  # Python ints: no overflow
        raise CorruptArtifactError(
            f"{file}: the data offsets of tensor {name} do not fit its shape"
        )
    if begin % itemsize:
        raise CorruptArtifactError(f"{file}: tensor {name} is not aligned to its type")
    return code, shape, begin, end


def _check_coverage(file: Path, entries: dict, start: int, size: int):
    """The tensors cover the data without gaps or overlaps, up to the end of the file."""
    position = 0
    for name, (_, _, begin, end) in sorted(
        entries.items(), key=lambda item: (item[1][2], item[1][3])
    ):
        if begin != position:
            raise CorruptArtifactError(
                f"{file}: the data of tensor {name} leaves a gap or overlaps"
            )
        position = end
    if start + position != size:
        raise CorruptArtifactError(f"{file} has {size} bytes, {start + position} expected")


class SafetensorsContainer:
    """All tensors in one ``.safetensors`` file: the container of models and of the hub.

    The format of safetensors (ARCHITECTURE.md, "safetensors 1"): an 8-byte little-endian
    header length, a JSON header with every tensor's dtype, shape and data offsets and the
    metadata in ``__metadata__``, then the data. Tensors are laid out by descending item
    size, then name, without gaps, so each is aligned to its type; the header is padded
    with spaces so that the data start at a multiple of 64 bytes.
    """

    name = "safetensors"
    version = 1
    suffix = ".safetensors"

    def write(
        self,
        path,
        tensors: Mapping[str, ndarray],
        metadata: Mapping[str, str],
        expected: Mapping[str, str] | None = None,
    ):
        """Write a new file; ``expected`` checksums, if given, must match the data.

        The data are hashed while they are written; the header, whose length does not
        depend on the checksums, is completed afterwards.
        """
        path = Path(path)
        if path.suffix != self.suffix:
            raise ValueError(f"{path} must end with {self.suffix}")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{path} exists; an artifact is never overwritten")
        # Snapshots: the header and the fingerprint use only these, so a caller that changes
        # its mapping, an array's shape or the metadata meanwhile cannot change the file.
        tensors = {n: a.view() if isinstance(a, ndarray) else a for n, a in tensors.items()}
        metadata = dict(metadata)
        codes = _check_for_writing(tensors, metadata)
        order, layout = _layout(tensors)
        length = self._reserved_length(tensors, codes, layout, metadata)
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.writing-", dir=path.parent)
        tmp = Path(name)  # created by this call, so only this call removes it
        try:
            with os.fdopen(descriptor, "wb") as f:
                f.write(b"\0" * (_SAFETENSORS_LENGTH + length))
                checksums = _write_data(f, tensors, order, expected)
                full = dict(metadata)
                for name, checksum in checksums.items():
                    full[CHECKSUM_PREFIX + name] = checksum
                entries = {
                    n: (codes[n], tuple(int(d) for d in tensors[n].shape), checksums[n])
                    for n in tensors
                }
                full[FINGERPRINT_KEY] = fingerprint(entries, full)
                header = self._header(tensors, codes, layout, full)
                if len(header) > length:  # pragma: no cover - the snapshots rule it out
                    raise RuntimeError("the header outgrew the space reserved for it")
                f.seek(0)
                f.write(length.to_bytes(_SAFETENSORS_LENGTH, "little"))
                f.write(header + b" " * (length - len(header)))
                f.flush()
                os.fsync(f.fileno())
            _publish(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        _sync_directory(path.parent)

    def _reserved_length(self, tensors, codes, layout, metadata) -> int:
        """The header length to reserve: the header with placeholder checksums and
        fingerprint of the same length, padded so the data start at ALIGNMENT."""
        placeholder = dict(metadata)
        for name in tensors:
            placeholder[CHECKSUM_PREFIX + name] = _hash.PLACEHOLDER
        placeholder[FINGERPRINT_KEY] = _hash.PLACEHOLDER
        length = len(self._header(tensors, codes, layout, placeholder))
        length += -(_SAFETENSORS_LENGTH + length) % ALIGNMENT
        if length > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"the header would exceed {MAX_SAFETENSORS_HEADER_BYTES} bytes")
        return length

    @staticmethod
    def _header(tensors, codes, layout, metadata) -> bytes:
        header: dict[str, Any] = {"__metadata__": dict(metadata)}
        for name, (begin, end) in layout.items():
            header[name] = {
                "dtype": codes[name],
                "shape": [int(d) for d in tensors[name].shape],
                "data_offsets": [begin, end],
            }
        return canonical_json(header).encode("utf-8")

    def open(self, path, *, mmap: bool) -> tuple[dict[str, ndarray], dict[str, str]]:
        path = Path(path)
        with _open_regular(path) as handle:
            before = _identity(handle)
            start, layout, metadata = _read_safetensors_header(handle, path)
            entries = _integrity(path, {n: (c, s) for n, (c, s, _, _) in layout.items()}, metadata)
            size = max((end for _, _, _, end in layout.values()), default=0)
            data = _read_data(handle, path, (start, size), mmap, before)
        tensors = {
            name: _open_tensor(path, name, spec, data, entries[name][2])
            for name, spec in layout.items()
        }
        return tensors, dict(metadata)

    def open_bytes(self, data: bytes, path) -> tuple[dict[str, ndarray], dict[str, str]]:
        """``open`` of a file given as its bytes (read once and checked by the caller,
        such as a file of the Hub); ``path`` names it in errors. Every checksum is
        checked."""
        path = Path(path)
        start, layout, metadata = _read_safetensors_header(io.BytesIO(data), path, len(data))
        entries = _integrity(path, {n: (c, s) for n, (c, s, _, _) in layout.items()}, metadata)
        view = memoryview(data)[start:]
        tensors = {
            name: _open_tensor(path, name, spec, view, entries[name][2])
            for name, spec in layout.items()
        }
        return tensors, dict(metadata)

    def verify(self, path):
        path = Path(path)
        with _open_regular(path) as handle:
            before = _identity(handle)
            start, layout, metadata = _read_safetensors_header(handle, path)
            entries = _integrity(path, {n: (c, s) for n, (c, s, _, _) in layout.items()}, metadata)
            for name, (_, _, begin, end) in layout.items():
                handle.seek(start + begin)
                digest = _hash.Hasher()
                remaining = end - begin
                while remaining:
                    chunk = handle.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise CorruptArtifactError(f"{path} ended early")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if digest.checksum() != entries[name][2]:
                    raise CorruptArtifactError(
                        f"tensor {name} of {path} does not match its checksum"
                    )
            _unchanged(handle, path, before)


def _layout(tensors: dict[str, ndarray]) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """The order of the tensors, by descending item size, then name, and their offsets."""
    order = sorted(tensors, key=lambda n: (-tensors[n].dtype.itemsize, n))
    layout = {}
    position = 0
    for name in order:
        layout[name] = (position, position + tensors[name].nbytes)
        position += tensors[name].nbytes
    return order, layout


def _write_data(f, tensors: dict[str, ndarray], order: list[str], expected) -> dict[str, str]:
    """Hash and write the data of every tensor in ``order``; return their checksums."""
    checksums = {}
    for name in order:
        digest = _hash.Hasher()
        array = tensors[name]
        if array.size:
            view = memoryview(array.reshape(-1).view(np.uint8))
            for begin in range(0, len(view), _CHUNK):
                # One immutable copy is hashed and written, so the file holds
                # exactly what its checksum describes even if the source changes.
                part = bytes(view[begin : begin + _CHUNK])
                digest.update(part)
                f.write(part)
        checksums[name] = digest.checksum()
        if expected is not None and expected.get(name) != checksums[name]:
            raise CorruptArtifactError(
                f"tensor {name} changed since it was checked; nothing was written"
            )
    return checksums


def _read_data(handle, path: Path, span: tuple[int, int], mmap: bool, before):
    """The data part at ``span`` (start, size): one map, None if empty, or the bytes, read
    whole and checked for a change of the file meanwhile."""
    start, size = span
    if mmap:
        if not size:
            return None
        return np.memmap(handle, dtype=np.uint8, mode="r", offset=start, shape=(size,))
    raw = handle.read(size)
    if len(raw) != size:
        raise CorruptArtifactError(f"{path} ended early")
    _unchanged(handle, path, before)
    return raw


def _open_tensor(path: Path, name: str, spec, data, checksum: str) -> ndarray:
    """Tensor ``name`` with ``spec`` (code, shape, begin, end) on the data of the file: a
    view of the map, or an array on the bytes after checking its checksum."""
    code, shape, begin, end = spec
    dtype = np.dtype(DTYPE_CODES[code])
    if begin == end:
        if checksum != _hash.EMPTY:
            raise CorruptArtifactError(f"tensor {name} of {path} does not match its checksum")
        array = np.empty(shape, dtype=dtype)
        array.setflags(write=False)
        return array
    if isinstance(data, np.memmap):
        return data[begin:end].view(dtype).reshape(shape)
    view = memoryview(data)[begin:end]
    digest = _hash.Hasher()
    for offset in range(0, len(view), _CHUNK):
        digest.update(view[offset : offset + _CHUNK])
    if digest.checksum() != checksum:
        raise CorruptArtifactError(f"tensor {name} of {path} does not match its checksum")
    return np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=begin).reshape(shape)


def _identity(handle) -> tuple[int, int]:
    info = os.fstat(handle.fileno())
    return info.st_size, info.st_mtime_ns


def _unchanged(handle, path: Path, before: tuple[int, int]):
    """Raise if the file changed size or time while it was read (another writer)."""
    if _identity(handle) != before:
        raise CorruptArtifactError(f"{path} changed while it was read")


def _move_write_through(tmp: Path, path: Path):  # pragma: no cover - Windows only
    """MoveFileExW with MOVEFILE_WRITE_THROUGH and without MOVEFILE_REPLACE_EXISTING."""
    import ctypes  # noqa: PLC0415 - Windows only

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    move = kernel32.MoveFileExW
    move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    move.restype = ctypes.c_int
    if not move(str(tmp), str(path), 0x8):
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        if error in (80, 183):  # ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS
            raise FileExistsError(f"{path} exists; an artifact is never overwritten")
        raise ctypes.WinError(error)  # type: ignore[attr-defined]


def _publish(tmp: Path, path: Path):
    """Move the finished ``tmp`` to ``path`` without ever replacing an existing file."""
    if os.name == "nt":  # pragma: no cover - tested on the Windows runners of CI
        _move_write_through(tmp, path)
        return
    os.link(tmp, path)  # fails with FileExistsError if path exists meanwhile
    tmp.unlink()
