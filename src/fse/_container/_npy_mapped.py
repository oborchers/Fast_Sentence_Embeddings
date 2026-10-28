#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Writing an NpyContainer artifact in place: a .npy file mapped by the workers that fill
it, and the exclusive publication of a finished artifact directory.

The ShardSink of fse._store writes the rows of a shard straight into its ``vectors.npy``:
the file is created with the header of its full shape and preallocated range by range,
so a full disk raises an OSError before a worker writes; a shard that ends early gets
the header of its rows, padded to the same length, and is truncated. ``write_manifest``
of NpyContainer then commits the directory, and ``publish_directory`` renames it into
place without ever replacing anything. Every step syncs as ``sync_file`` does: on macOS
with F_FULLFSYNC, since fsync there does not reach the drive.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import io
import mmap
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.lib import format as npy_format

from ._common import ALIGNMENT, _sync_directory

# fcntl for F_FULLFSYNC; Windows has none.
_fcntl: ModuleType | None = None if os.name == "nt" else importlib.import_module("fcntl")

_ZEROS = bytes(1 << 20)


def header(dtype: np.dtype, shape: tuple[int, ...], length: int | None = None) -> bytes:
    """The .npy header (version 1.0) of a C-order array, padded with spaces to ``length``
    bytes (by default the shortest multiple of ALIGNMENT), ending in a newline."""
    buffer = io.BytesIO()
    npy_format.write_array_header_1_0(
        buffer,
        {
            "descr": npy_format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        },
    )
    raw = buffer.getvalue()  # magic, version, 2-byte length, the dict, spaces, newline
    length = len(raw) if length is None else length
    if len(raw) > length or length % ALIGNMENT or length - 10 > 0xFFFF:
        raise ValueError(f"a .npy header of {len(raw)} bytes does not fit in {length} bytes")
    body = raw[10:].rstrip(b" \n")
    return raw[:8] + (length - 10).to_bytes(2, "little") + body.ljust(length - 11) + b"\n"


def create(file: Path, dtype: np.dtype, shape: tuple[int, ...]) -> tuple[int, int]:
    """A new file ``file`` (never an existing one) with the header of ``shape`` and the
    size of all its data, not yet allocated. Returns the descriptor and the data offset."""
    head = header(dtype, shape)
    nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(file, flags, 0o644)
    try:
        os.write(fd, head)
        os.ftruncate(fd, len(head) + nbytes)
    except BaseException:
        os.close(fd)
        raise
    return fd, len(head)


def allocate(fd: int, begin: int, end: int) -> None:
    """Allocate the bytes [begin, end) of the file on disk, so writing them through a
    mapping cannot fail for lack of space: posix_fallocate where the system has it, else
    zeros written in place. A full disk raises an OSError here."""
    if end <= begin:
        return
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is not None and sys.platform.startswith("linux"):
        try:
            fallocate(fd, begin, end - begin)
            return
        except OSError as err:
            if err.errno not in (95, 22):  # EOPNOTSUPP, EINVAL: the filesystem cannot
                raise
    position = begin
    while position < end:
        position += _write_at(fd, _ZEROS[: min(len(_ZEROS), end - position)], position)


def _write_at(fd: int, data: bytes, position: int) -> int:
    """os.pwrite, or a seek and a write where it is missing (Windows)."""
    if hasattr(os, "pwrite"):
        return os.pwrite(fd, data, position)
    os.lseek(fd, position, os.SEEK_SET)  # pragma: no cover - Windows only
    return os.write(fd, data)  # pragma: no cover - Windows only


def map_range(fd: int, offset: int, begin: int, end: int, dtype: np.dtype, shape: tuple[int, ...]):
    """A writable mapping of the data bytes [begin, end) (from the data offset) and the
    array of ``shape`` over it. Close the mapping only when no view of the array is left."""
    start = offset + begin
    aligned = start - start % mmap.ALLOCATIONGRANULARITY
    mapping = mmap.mmap(fd, offset + end - aligned, offset=aligned, access=mmap.ACCESS_WRITE)
    array = np.frombuffer(
        mapping, dtype=dtype, count=int(np.prod(shape, dtype=np.int64)), offset=start - aligned
    )
    return mapping, array.reshape(shape)


def shrink(fd: int, offset: int, dtype: np.dtype, shape: tuple[int, ...]) -> None:
    """Give the file the header of its first rows ``shape``, padded to the same data
    offset, and cut it after their data. No mapping of the file may be open."""
    _write_at(fd, header(dtype, shape, offset), 0)
    os.ftruncate(fd, offset + int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize)


def sync_file(fd: int) -> None:
    """Make the file's data durable: F_FULLFSYNC on macOS, where fsync only reaches the
    drive's cache; fsync elsewhere (FlushFileBuffers on Windows)."""
    if _fcntl is not None and hasattr(_fcntl, "F_FULLFSYNC"):
        _fcntl.fcntl(fd, _fcntl.F_FULLFSYNC)
        return
    os.fsync(fd)


def publish_directory(tmp: Path, path: Path) -> None:
    """Rename the finished directory ``tmp`` to ``path`` without ever replacing anything,
    not even an empty directory, then sync the parent. Raises FileExistsError if ``path``
    exists, and OSError where the system has no exclusive rename of directories."""
    _rename_exclusive(tmp, path)
    _sync_directory(path.parent)


def _rename_exclusive(tmp: Path, path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - tested on the Windows runners of CI
        _move_no_replace(tmp, path)
        return
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    if sys.platform == "darwin":
        result = libc.renamex_np(
            os.fsencode(tmp), os.fsencode(path), ctypes.c_uint(0x4)
        )  # RENAME_EXCL
    elif hasattr(libc, "renameat2"):
        at_cwd = ctypes.c_int(-100)
        result = libc.renameat2(
            at_cwd, os.fsencode(tmp), at_cwd, os.fsencode(path), ctypes.c_uint(1)
        )  # RENAME_NOREPLACE
    else:  # pragma: no cover - a system without an exclusive rename
        raise OSError(
            f"this system cannot rename {tmp} without replacing; the store needs Linux, macOS or Windows"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in (17, 39, 66):  # EEXIST, ENOTEMPTY (Linux), ENOTEMPTY (macOS)
            raise FileExistsError(f"{path} exists; a shard is never replaced")
        raise OSError(error, os.strerror(error), str(path))


def _move_no_replace(tmp: Path, path: Path) -> None:  # pragma: no cover - Windows only
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    move = kernel32.MoveFileExW
    move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    move.restype = ctypes.c_int
    if not move(str(tmp), str(path), 0):  # without MOVEFILE_REPLACE_EXISTING
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        if error in (80, 183):  # ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS
            raise FileExistsError(f"{path} exists; a shard is never replaced")
        raise ctypes.WinError(error)  # type: ignore[attr-defined]
