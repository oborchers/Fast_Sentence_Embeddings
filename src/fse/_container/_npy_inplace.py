#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Writing an NpyContainer artifact in place: a .npy file written at the offsets of its
rows, and the exclusive publication of a finished artifact directory.

The ShardSink of fse._store writes the rows of a shard straight into its ``vectors.npy``:
the file is created with the header of its full shape, the rows are written at their
offsets as they are committed (a full disk raises an OSError there), and a shard that
ends early gets the header of its rows, padded to the same length, and is truncated. ``write_manifest``
of NpyContainer then commits the directory, and ``publish_directory`` renames it into
place without ever replacing anything.

Durability on macOS, where fsync only reaches the drive's cache, needs F_FULLFSYNC, which
flushes that whole cache and costs milliseconds. A shard therefore sends its files and
its manifest to the drive with plain fsyncs (``sync_data``), makes all of them durable
with one full sync of its directory before the rename, and makes the rename durable with
a second one of the store (``publish_directory``). Elsewhere every step is an fsync.
``sync_file`` makes a single file durable on its own.
"""

from __future__ import annotations

import ctypes
import io
import os
import sys
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from . import _common
from ._common import ALIGNMENT, PublishedNotDurableError, _sync_directory


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
    size of all its data (sparse until written). Returns the descriptor and the offset of
    the data."""
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
        write_at(fd, memoryview(head), 0)
        os.ftruncate(fd, len(head) + nbytes)
    except BaseException:
        os.close(fd)
        raise
    return fd, len(head)


def _seek_and_write(fd: int, data: memoryview, position: int) -> int:
    """os.pwrite where it is missing (Windows); a file is written by one thread at a time."""
    os.lseek(fd, position, os.SEEK_SET)
    return os.write(fd, data)


_write_some = getattr(os, "pwrite", _seek_and_write)


def write_at(fd: int, data: memoryview, position: int) -> None:
    """Write all of ``data`` at ``position``: a write may write less, so it loops. A full
    disk raises an OSError."""
    view = memoryview(data).cast("B")
    while len(view):
        n = _write_some(fd, view, position)
        if n <= 0:
            raise OSError(f"a write at byte {position} made no progress")
        view, position = view[n:], position + n


def shrink(fd: int, offset: int, dtype: np.dtype, shape: tuple[int, ...]) -> None:
    """Give the file the header of its first rows ``shape``, padded to the same data
    offset, and cut it after their data."""
    write_at(fd, memoryview(header(dtype, shape, offset)), 0)
    os.ftruncate(fd, offset + int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize)


def sync_file(fd: int) -> None:
    """Make the file's data durable: F_FULLFSYNC on macOS, where fsync only reaches the
    drive's cache; fsync elsewhere (FlushFileBuffers on Windows)."""
    fcntl = _common._fcntl
    if fcntl is not None and hasattr(fcntl, "F_FULLFSYNC"):
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        return
    os.fsync(fd)


def sync_data(fd: int) -> None:
    """Send the file's data to the drive (fsync): durable elsewhere, and on macOS once a
    full sync follows, as ``publish_directory`` and the barrier before it do."""
    os.fsync(fd)


def publish_directory(tmp: Path, path: Path) -> None:
    """Rename the finished directory ``tmp`` to ``path`` without ever replacing anything,
    not even an empty directory, then sync the parent. Raises FileExistsError if ``path``
    exists, and OSError where the system has no exclusive rename of directories; nothing
    is published then. A failed sync after the rename raises PublishedNotDurableError: the
    directory is published at ``path`` and stays there, but its rename is not confirmed
    durable."""
    _rename_exclusive(tmp, path)
    try:
        _sync_directory(path.parent)
    except OSError as error:
        raise PublishedNotDurableError(
            f"{path} is published, but syncing {path.parent} failed ({error}); the rename is "
            "not confirmed durable"
        ) from error


def _rename_exclusive(tmp: Path, path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - tested on the Windows runners of CI
        _move_no_replace(tmp, path)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = libc.renamex_np(
            os.fsencode(tmp), os.fsencode(path), ctypes.c_uint(0x4)
        )  # RENAME_EXCL
    elif sys.platform == "linux":
        result = _rename_linux(libc, tmp, path)
    else:  # pragma: no cover - a system without an exclusive rename
        raise OSError(
            f"this system cannot rename {tmp} without replacing; the store needs Linux, macOS or Windows"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in (17, 39, 66):  # EEXIST, ENOTEMPTY (Linux), ENOTEMPTY (macOS)
            raise FileExistsError(f"{path} exists; it is never replaced")
        raise OSError(error, os.strerror(error), str(path))


def _rename_linux(libc: ctypes.CDLL, tmp: Path, path: Path) -> int:
    """musl has syscall but no renameat2 wrapper; both use RENAME_NOREPLACE."""
    args = (
        ctypes.c_int(-100),
        os.fsencode(tmp),
        ctypes.c_int(-100),
        os.fsencode(path),
        ctypes.c_uint(1),
    )
    if hasattr(libc, "renameat2"):
        return int(libc.renameat2(*args))
    # Linux UAPI: arch/x86/entry/syscalls/syscall_64.tbl (316),
    # include/uapi/asm-generic/unistd.h (276, used by aarch64).
    # https://github.com/torvalds/linux/blob/v6.17/arch/x86/entry/syscalls/syscall_64.tbl
    # https://github.com/torvalds/linux/blob/v6.17/include/uapi/asm-generic/unistd.h
    number = {"x86_64": 316, "aarch64": 276}.get(os.uname().machine)
    if number is None or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError("this Linux architecture has no supported exclusive rename syscall")
    call = libc.syscall
    call.argtypes = (
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    call.restype = ctypes.c_long
    return int(call(ctypes.c_long(number), *args))


def _move_no_replace(tmp: Path, path: Path) -> None:  # pragma: no cover - Windows only
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    move = kernel32.MoveFileExW
    move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    move.restype = ctypes.c_int
    # MOVEFILE_WRITE_THROUGH (0x8), without MOVEFILE_REPLACE_EXISTING: returns once the
    # move is on the disk.
    if not move(str(tmp), str(path), 0x8):
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        if error in (80, 183):  # ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS
            raise FileExistsError(f"{path} exists; it is never replaced")
        raise ctypes.WinError(error)  # type: ignore[attr-defined]
