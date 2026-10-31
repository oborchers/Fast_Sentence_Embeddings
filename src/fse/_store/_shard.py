#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""One shard being written: rows reserved for the workers, committed in order, then
published whole (docs/architecture/store.md, "sentence_vectors 1").

A shard is written in a temporary directory ``.shard-<first row>.writing-<hex>`` of the
store. The worker of a task writes its rows into a buffer of the task (the window of the
runtime bounds them); its commit writes them into ``vectors.npy`` at their offset
(``pwrite``, so a full disk raises an OSError there) and feeds the checksum of the vectors,
in row order. The scales and inverse norms are kept in memory. Finishing shrinks a shard
that ends early, syncs the file, writes the scales, the inverse norms and the manifest,
and renames the directory into place without replacing anything. Until then the rows are
not durable; ``abort`` removes the temporary directory.

Writing through a mapping of the file instead (the first design) cost about half the
throughput: page faults on fresh file pages and a mapping per task (``bench_store.py``,
M1 Pro, int8 at 10 workers: 1.04 against 1.96 million rows per second).
"""

from __future__ import annotations

import os
import secrets
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy import ndarray
from numpy.lib import format as npy_format

from fse import _codes
from fse._container import Hasher, checksum, remove_temporary, tensor_code
from fse._container._npy import write_manifest
from fse._container._npy_inplace import create, publish_directory, shrink, sync_data, write_at

from ._layout import StoreInfo, shard_metadata, shard_name


class ShardWriter:
    """The shard of the rows ``first_row`` to ``first_row + capacity - 1`` of a store."""

    def __init__(
        self, store: Path, info: StoreInfo, first_row: int, capacity: int, created_by: str
    ):
        self.store, self.info, self.first_row, self.capacity = store, info, first_row, capacity
        self.created_by = created_by
        self.dtype = np.dtype(_codes.STORAGE_TYPES[info.dtype])
        self.row_bytes = info.dim * self.dtype.itemsize
        self.tmp = store / f".{shard_name(first_row)}.writing-{secrets.token_hex(4)}"
        self.tmp.mkdir()
        # vectors.npy is created at the first write, so the shards reserved ahead of their
        # writes (a task over many small shards) hold no open file.
        self.fd, self.offset = -1, 0
        self.scales = np.empty(capacity, np.float32) if info.dtype == "int8" else None
        self.norms = np.empty(capacity, np.float32)
        self.reserved = self.committed = self.written = 0
        self._written_rows = threading.Condition()
        # Rows handed to the writer thread of the sink to commit; kept by the calling thread,
        # while ``committed`` is kept by the thread that commits.
        self.queued = 0
        self.end = capacity  # sealed earlier by seal()
        self.digest = Hasher()
        self._open: dict[int, ndarray] = {}  # first row in the shard -> the codes reserved

    @property
    def full(self) -> bool:
        return self.reserved == self.end

    @property
    def done(self) -> bool:
        return self.committed == self.end

    def reserve(self, n_rows: int) -> tuple[ndarray, ndarray | None, ndarray]:
        """Codes (a buffer of the task), scales and inverse norms of the next ``n_rows``."""
        begin, end = self.reserved, self.reserved + n_rows
        if end > self.end:
            raise ValueError(f"rows {begin} to {end} beyond the {self.end} rows of the shard")
        codes = np.empty((n_rows, self.info.dim), self.dtype)
        self._open[begin] = codes
        self.reserved = end
        scales = None if self.scales is None else self.scales[begin:end]
        return codes, scales, self.norms[begin:end]

    def commit(self, begin: int, codes: ndarray) -> None:
        """The rows from ``begin`` (in the shard), which must be the codes reserved there,
        are finished: written into the file and hashed."""
        if begin != self.committed or begin not in self._open:
            raise ValueError(
                f"rows from {begin} of the shard committed, expected rows from {self.committed}"
            )
        if codes is not self._open[begin]:
            raise ValueError(f"the rows from {begin} are not the target reserved for them")
        self.digest.update(codes)
        del self._open[begin]  # only once the commit is checked
        self.committed += len(codes)

    def _file(self) -> None:
        """Create vectors.npy with the header of the full shard, once."""
        if self.fd < 0:
            self.fd, self.offset = create(
                self.tmp / "vectors.npy", self.dtype, (self.capacity, self.info.dim)
            )

    def write(self, begin: int, codes: ndarray) -> None:
        """Write the committed rows from ``begin`` (in the shard) into the file; positional,
        so it may run in another thread than ``commit`` (only one thread writes)."""
        self._file()
        write_at(self.fd, memoryview(codes).cast("B"), self.offset + begin * self.row_bytes)
        with self._written_rows:
            self.written += len(codes)
            self._written_rows.notify_all()

    def wait_written(self, failed: Callable[[], bool]) -> None:
        """Wait until every committed row is written; raise if the writing thread failed."""
        with self._written_rows:
            while self.written < self.committed:
                if failed():
                    raise RuntimeError(
                        "the rows of the shard were not written; it is not published"
                    )
                self._written_rows.wait(0.05)

    @property
    def queued_all(self) -> bool:
        """Whether every row of the shard was handed over to be committed."""
        return self.queued == self.end

    def drop_uncommitted(self) -> None:
        """End the shard at its committed rows: the reserved rows after them are dropped."""
        self._open.clear()
        self.reserved = self.end = self.queued = self.committed

    def seal(self) -> None:
        """End the shard at the rows reserved so far."""
        self.end = self.reserved

    def finish(self, failed: Callable[[], bool] = lambda: False) -> int:
        """Publish the committed rows (all reserved rows must be committed) once they are
        written (``failed`` says whether the writing stopped); returns them."""
        if self.committed != self.reserved:
            raise RuntimeError(f"{self.reserved - self.committed} reserved rows were not committed")
        self.wait_written(failed)
        self._file()
        rows = self.committed
        if rows != self.capacity:
            shrink(self.fd, self.offset, self.dtype, (rows, self.info.dim))
        sync_data(self.fd)
        os.close(self.fd)
        self.fd = -1
        code = tensor_code(np.empty(0, self.dtype))
        entries = {"vectors": (code, (rows, self.info.dim), self.digest.checksum())}
        extra = {"inv_norms": self.norms[:rows]}
        if self.scales is not None:
            extra["scales"] = self.scales[:rows]
        for name, array in extra.items():
            entries[name] = (
                tensor_code(array),
                array.shape,
                _write(self.tmp / f"{name}.npy", array),
            )
        metadata = shard_metadata(self.info, self.first_row, rows, self.created_by)
        manifest_tmp = self.tmp / f".manifest.json.writing-{secrets.token_hex(4)}"
        # Two full syncs on macOS: the directory before the rename makes the files and
        # the manifest durable, the store after it the rename (docs/architecture/store.md,
        # "sentence_vectors 1").
        write_manifest(self.tmp, entries, metadata, manifest_tmp, sync_data, staged=True)
        publish_directory(self.tmp, self.store / shard_name(self.first_row))
        # A long stream keeps no arrays of published shards.
        self.norms = np.empty(0, np.float32)
        self.scales = None if self.scales is None else np.empty(0, np.float32)
        return rows

    def abort(self) -> OSError | None:
        """Drop the shard: removes the temporary directory; returns the error of a removal
        that failed (None once it is gone, also if it was gone already)."""
        self._open.clear()
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        return remove_temporary(self.tmp)


def _write(file: Path, array: ndarray) -> str:
    """A small tensor as a .npy file sent to the drive; returns the checksum of its data."""
    data = np.ascontiguousarray(array)
    fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    with os.fdopen(fd, "wb") as f:
        npy_format.write_array_header_1_0(f, npy_format.header_data_from_array_1_0(data))
        f.write(data.tobytes())
        f.flush()
        sync_data(f.fileno())
    return checksum(data)
