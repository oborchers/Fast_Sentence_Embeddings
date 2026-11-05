"""The tensors of a shard as columns: streamed to a file, kept in memory, or constant.

A shard is a directory of one ``.npy`` file per tensor and a manifest. How a tensor
gets there depends on its size and on who knows its values:

- ``StreamColumn`` is the data of a tensor that grows with the rows. Its file is created
  at the first write with the header of the full capacity; each commit writes its rows at
  their offset (``pwrite``, so a full disk raises an OSError there) and feeds the
  checksum, in row order. Finishing cuts the file to the rows committed.
- ``MemoryColumn`` is a small tensor that the workers fill through views; it is written
  once, when the shard is published.
- ``ConstantColumn`` is a tensor without rows, the same in every shard; it is written once
  and not cut when the shard ends early.

Every ``finish`` returns the manifest entry of its tensor: the code, the shape and the
checksum of the data. The order in which the columns finish is the order of the syncs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy import ndarray
from numpy.lib import format as npy_format

from fse._container import Hasher, checksum, tensor_code
from fse._container._npy_inplace import create, shrink, sync_data, write_at

Entry = tuple[str, tuple[int, ...], str]


class StreamColumn:
    """A positional file and an ordered digest; IO completion is owned by the shard."""

    def __init__(
        self,
        directory: Path,
        name: str,
        dtype: np.dtype,
        row_shape: tuple[int, ...],
        capacity: int,
        *,
        always: bool = False,
    ):
        self.name, self.dtype, self.row_shape, self.capacity = name, dtype, row_shape, capacity
        self.always = always  # the file exists even if no row was written
        self.path = directory / f"{name}.npy"
        self.row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
        self.fd, self.offset = -1, 0
        self.digest = Hasher()

    def ensure(self) -> None:
        """Create the file with the header of the full shard, once."""
        if self.fd < 0:
            self.fd, self.offset = create(self.path, self.dtype, (self.capacity, *self.row_shape))

    def write(self, begin: int, rows: ndarray) -> None:
        """Write the rows from ``begin`` (in the shard) into the file."""
        self.ensure()
        write_at(self.fd, memoryview(rows).cast("B"), self.offset + begin * self.row_bytes)

    def commit(self, rows: ndarray) -> None:
        """Feed the checksum with the next rows."""
        self.digest.update(rows)

    def finish(self, rows: int, *, synced: bool) -> Entry:
        """Cut the file to ``rows``, send it to the drive if ``synced`` and close it."""
        if self.always:
            self.ensure()
        if self.fd < 0:
            raise RuntimeError(f"{self.name} were not written; the shard is not published")
        if rows != self.capacity:
            shrink(self.fd, self.offset, self.dtype, (rows, *self.row_shape))
        if synced:
            sync_data(self.fd)
        self.close()
        return tensor_code(np.empty(0, self.dtype)), (rows, *self.row_shape), self.digest.checksum()

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class MemoryColumn:
    """One value per row, kept in memory until the shard is published."""

    def __init__(self, directory: Path, name: str, dtype: np.dtype, capacity: int):
        self.directory, self.name, self.dtype = directory, name, dtype
        self.array = np.empty(capacity, dtype)

    def view(self, begin: int, end: int) -> ndarray:
        """The values of the rows ``begin`` to ``end``, which a worker fills."""
        return self.array[begin:end]

    def finish(self, rows: int, *, synced: bool) -> Entry:
        """Write the first ``rows`` values as ``<name>.npy`` and release the memory."""
        array = self.array[:rows]
        entry = write_tensor(self.directory, self.name, array, synced=synced)
        self.array = np.empty(0, self.dtype)  # a long stream keeps no arrays of published shards
        return entry


class ConstantColumn:
    """A tensor that does not depend on the rows: the same in every shard, never cut."""

    def __init__(self, directory: Path, name: str, array: ndarray):
        self.directory, self.name, self.array = directory, name, array

    def finish(self, rows: int, *, synced: bool) -> Entry:
        del rows  # a shard that ends early holds the whole tensor
        return write_tensor(self.directory, self.name, self.array, synced=synced)


def write_tensor(directory: Path, name: str, array: ndarray, *, synced: bool) -> Entry:
    """A small tensor as ``<name>.npy`` sent to the drive; returns its manifest entry."""
    data = np.ascontiguousarray(array)
    fd = os.open(
        directory / f"{name}.npy",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    with os.fdopen(fd, "wb") as f:
        npy_format.write_array_header_1_0(f, npy_format.header_data_from_array_1_0(data))
        f.write(data.tobytes())
        f.flush()
        if synced:
            sync_data(f.fileno())
    return tensor_code(array), array.shape, checksum(data)
