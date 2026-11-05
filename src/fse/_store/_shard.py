#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""One shard being written: rows reserved for the workers, committed in order, then
published whole (docs/architecture/store.md, "sentence_vectors 1").

A shard is written in a temporary directory ``.shard-<first row>.writing-<hex>`` of the
store. The worker of a task writes its rows into a buffer of the task (the window of the
runtime bounds them); its commit writes them into the files of the streamed columns at their
offset (``pwrite``, so a full disk raises an OSError there) and feeds their checksums, in row
order. Small columns are kept in memory. Finishing shrinks a shard that ends early, writes the
small columns and the manifest, and renames the directory into place without replacing
anything. Until then the rows are not published; ``abort`` removes the temporary directory.
With ``synced=True``, every file and the directory entries are synced before publication is
confirmed.

``ShardBase`` is what every kind of store shares: the order of reservations, commits and
writes, the end of a shard and its publication through a list of columns. ``ShardWriter``
is the shard of the sentence vectors: ``vectors``, optional ``doc_ids``, ``inv_norms`` and
for int8 ``scales``.

Writing through a mapping of the file instead (the first design) cost about half the
throughput: page faults on fresh file pages and a mapping per task (``bench_store.py``,
M1 Pro, int8 at 10 workers: 1.04 against 1.96 million rows per second).
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._container import remove_temporary
from fse._container._npy import write_manifest
from fse._container._npy_inplace import publish_directory, sync_data

from ._columns import Entry, MemoryColumn, StreamColumn
from ._id_writer import IdColumn
from ._layout import StoreInfo, shard_metadata, shard_name


class ShardBase:
    """The shard of the rows ``first_row`` to ``first_row + capacity - 1`` of a store."""

    def __init__(
        self, store: Path, first_row: int, capacity: int, created_by: str, *, synced: bool = True
    ):
        self.store, self.first_row, self.capacity = store, first_row, capacity
        self.created_by, self.synced = created_by, synced
        self.tmp = store / f".{shard_name(first_row)}.writing-{secrets.token_hex(4)}"
        self.tmp.mkdir()
        self.columns: list = []  # in the order they are finished: the order of the syncs
        self.reserved = self.committed = self.written = 0
        self._written_rows = threading.Condition()
        # Rows handed to the writer thread of the sink to commit; kept by the calling thread,
        # while ``committed`` is kept by the thread that commits.
        self.queued = 0
        self.end = capacity  # sealed earlier by seal()
        self._open: dict[int, ndarray] = {}  # first row in the shard -> the buffer reserved

    @property
    def full(self) -> bool:
        return self.reserved == self.end

    @property
    def done(self) -> bool:
        return self.committed == self.end

    def _reserved_rows(self, n_rows: int) -> int:
        """The first row of the next ``n_rows`` (in the shard), which must fit."""
        begin, end = self.reserved, self.reserved + n_rows
        if end > self.end:
            raise ValueError(f"rows {begin} to {end} beyond the {self.end} rows of the shard")
        return begin

    def _check_commit(self, begin: int, buffer: ndarray) -> None:
        """The rows from ``begin`` must be the next to commit and the buffer reserved there."""
        if begin != self.committed or begin not in self._open:
            raise ValueError(
                f"rows from {begin} of the shard committed, expected rows from {self.committed}"
            )
        if buffer is not self._open[begin]:
            raise ValueError(f"the rows from {begin} are not the target reserved for them")

    def reserve(self, n_rows: int) -> tuple:
        """The buffers of the next ``n_rows``, which the workers fill."""
        raise NotImplementedError

    def _committed_rows(self, begin: int, n_rows: int) -> None:
        del self._open[begin]  # only once the commit is checked
        self.committed += n_rows

    def _written_rows_done(self, n_rows: int) -> None:
        with self._written_rows:
            self.written += n_rows
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

    def _metadata(self, rows: int) -> dict[str, str]:
        raise NotImplementedError

    def finish(self, failed: Callable[[], bool] = lambda: False) -> int:
        """Publish the committed rows (all reserved rows must be committed) once they are
        written (``failed`` says whether the writing stopped); returns them."""
        if self.committed != self.reserved:
            raise RuntimeError(f"{self.reserved - self.committed} reserved rows were not committed")
        self.wait_written(failed)
        rows = self.committed
        entries: dict[str, Entry] = {
            column.name: column.finish(rows, synced=self.synced) for column in self.columns
        }
        manifest_tmp = self.tmp / f".manifest.json.writing-{secrets.token_hex(4)}"
        # Two full syncs on macOS: the directory before the rename makes the files and
        # the manifest durable, the store after it the rename (docs/architecture/store.md,
        # "sentence_vectors 1").
        write_manifest(
            self.tmp,
            entries,
            self._metadata(rows),
            manifest_tmp,
            sync_data if self.synced else lambda _fd: None,
            staged=True,
            directories=self.synced,
        )
        publish_directory(
            self.tmp, self.store / shard_name(self.first_row), sync_parent=self.synced
        )
        return rows

    def abort(self) -> OSError | None:
        """Drop the shard: removes the temporary directory; returns the error of a removal
        that failed (None once it is gone, also if it was gone already)."""
        self._open.clear()
        for column in self.columns:
            if isinstance(column, StreamColumn):
                column.close()
        return remove_temporary(self.tmp)


class ShardWriter(ShardBase):
    """A shard of sentence vectors: the codes, the inverse norms, for int8 the scales and
    optionally the document IDs."""

    def __init__(
        self,
        store: Path,
        info: StoreInfo,
        first_row: int,
        capacity: int,
        created_by: str,
        *,
        synced: bool = True,
    ):
        super().__init__(store, first_row, capacity, created_by, synced=synced)
        self.info = info
        self.dtype = np.dtype(_codes.STORAGE_TYPES[info.dtype])
        # vectors.npy is created at the first write, so the shards reserved ahead of their
        # writes (a task over many small shards) hold no open file.
        self.vectors = StreamColumn(
            self.tmp, "vectors", self.dtype, (info.dim,), capacity, always=True
        )
        self.ids = IdColumn(self.tmp, capacity) if info.has_doc_ids else None
        self.norms = MemoryColumn(self.tmp, "inv_norms", np.dtype(np.float32), capacity)
        self.scales = (
            MemoryColumn(self.tmp, "scales", np.dtype(np.float32), capacity)
            if info.dtype == "int8"
            else None
        )
        self.columns = [self.vectors, self.ids, self.norms, self.scales]
        self.columns = [column for column in self.columns if column is not None]

    def reserve(self, n_rows: int) -> tuple[ndarray, ndarray | None, ndarray]:
        """Codes (a buffer of the task), scales and inverse norms of the next ``n_rows``."""
        begin = self._reserved_rows(n_rows)
        end = begin + n_rows
        codes = np.empty((n_rows, self.info.dim), self.dtype)
        self._open[begin] = codes
        self.reserved = end
        scales = None if self.scales is None else self.scales.view(begin, end)
        return codes, scales, self.norms.view(begin, end)

    def commit(self, begin: int, codes: ndarray, doc_ids: ndarray | None = None) -> None:
        """The rows from ``begin`` (in the shard), which must be the codes reserved there,
        are finished: written into the file and hashed."""
        self._check_commit(begin, codes)
        if self.ids is not None:
            if doc_ids is None or len(doc_ids) != len(codes):
                raise ValueError("missing document IDs for committed rows")
            self.ids.commit(doc_ids)
        self.vectors.commit(codes)
        self._committed_rows(begin, len(codes))

    def write(self, begin: int, codes: ndarray, doc_ids: ndarray | None = None) -> None:
        """Write the committed rows from ``begin`` (in the shard) into the file; positional,
        so it may run in another thread than ``commit`` (only one thread writes)."""
        self.vectors.write(begin, codes)
        if self.ids is not None:
            if doc_ids is None:
                raise ValueError("missing document IDs for written rows")
            self.ids.write(begin, doc_ids)
        self._written_rows_done(len(codes))

    def _metadata(self, rows: int) -> dict[str, str]:
        metadata = shard_metadata(
            self.info, self.first_row, rows, self.created_by, unsynced=not self.synced
        )
        if self.ids is not None:
            metadata.update(self.ids.stats.metadata())
        return metadata
