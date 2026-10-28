#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""One shard being written: rows reserved and mapped for the workers, committed in order,
then published whole (ARCHITECTURE.md, "sentence_vectors 1").

A shard is written in a temporary directory ``.shard-<first row>.writing-<hex>`` of the
store: ``vectors.npy`` has the header of the full shard and is allocated and mapped range
by range as rows are reserved; the scales and inverse norms are kept in memory. Every
commit feeds the sha256 of the vectors (they arrive in row order). Finishing closes the
mappings (no view of them may be left), shrinks a shard that ends early, syncs the file,
writes the scales, the inverse norms and the manifest, and renames the directory into
place without replacing anything. Until then the rows are not durable; ``abort`` removes
the temporary directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from numpy import ndarray
from numpy.lib import format as npy_format

from fse import _codes
from fse._container import tensor_code
from fse._container._npy import write_manifest
from fse._container._npy_mapped import (
    allocate,
    create,
    map_range,
    publish_directory,
    shrink,
    sync_file,
)

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
        self.fd = -1
        try:
            self.fd, self.offset = create(
                self.tmp / "vectors.npy", self.dtype, (capacity, info.dim)
            )
        except BaseException:
            shutil.rmtree(self.tmp, ignore_errors=True)
            raise
        self.scales = np.empty(capacity, np.float32) if info.dtype == "int8" else None
        self.norms = np.empty(capacity, np.float32)
        self.reserved = self.committed = 0
        self.end = capacity  # sealed earlier by seal()
        self.digest = hashlib.sha256()
        self._maps: deque = deque()  # (end row, mapping) of committed ranges, to close
        self._open: dict[int, tuple] = {}  # first row in the shard -> (mapping, codes)

    @property
    def full(self) -> bool:
        return self.reserved == self.end

    @property
    def done(self) -> bool:
        return self.committed == self.end

    def reserve(self, n_rows: int) -> tuple[ndarray, ndarray | None, ndarray]:
        """Codes (mapped), scales and inverse norms of the next ``n_rows`` rows. A full
        disk raises an OSError before anything is mapped."""
        begin, end = self.reserved, self.reserved + n_rows
        if end > self.end:
            raise ValueError(f"rows {begin} to {end} beyond the {self.end} rows of the shard")
        allocate(self.fd, self.offset + begin * self.row_bytes, self.offset + end * self.row_bytes)
        mapping, codes = map_range(
            self.fd,
            self.offset,
            begin * self.row_bytes,
            end * self.row_bytes,
            self.dtype,
            (n_rows, self.info.dim),
        )
        self._open[begin] = (mapping, codes)
        self.reserved = end
        scales = None if self.scales is None else self.scales[begin:end]
        return codes, scales, self.norms[begin:end]

    def commit(self, begin: int, codes: ndarray) -> None:
        """The rows from ``begin`` (in the shard), which must be the codes reserved there,
        are finished."""
        if begin != self.committed or begin not in self._open:
            raise ValueError(
                f"rows from {begin} of the shard committed, expected rows from {self.committed}"
            )
        mapping, reserved = self._open[begin]
        if codes is not reserved and not (
            isinstance(codes, ndarray)
            and codes.shape == reserved.shape
            and codes.__array_interface__["data"][0] == reserved.__array_interface__["data"][0]
        ):
            raise ValueError(f"the rows from {begin} are not the target reserved for them")
        del self._open[begin]  # only once the commit is checked
        self.digest.update(memoryview(np.ascontiguousarray(codes)).cast("B"))
        self.committed += len(codes)
        del reserved, codes
        self._maps.append(mapping)
        self.release()

    def release(self) -> None:
        """Close the mappings of committed rows that no view holds any more, so the pages
        of finished rows leave the process."""
        while self._maps:
            try:
                self._maps[0].close()
            except BufferError:  # a view is still held (the executor's last item)
                return
            self._maps.popleft()

    def can_finish(self) -> bool:
        """Whether every row is committed and no view of the mappings is left."""
        if not self.done:
            return False
        self.release()
        return not self._maps

    def drop_uncommitted(self) -> None:
        """End the shard at its committed rows: the reserved rows after them are dropped
        (their views must be gone)."""
        for begin in [b for b in self._open if b >= self.committed]:
            mapping, _ = self._open.pop(begin)
            self._maps.append(mapping)
        self.reserved = self.end = self.committed

    def seal(self) -> None:
        """End the shard at the rows reserved so far."""
        self.end = self.reserved

    def finish(self) -> int:
        """Publish the committed rows (all reserved rows must be committed); returns them."""
        if self.committed != self.reserved:
            raise RuntimeError(f"{self.reserved - self.committed} reserved rows were not committed")
        self.release()
        if self._maps:
            raise RuntimeError(
                "a view of the rows of the shard is still in use; it cannot be published"
            )
        rows = self.committed
        if rows != self.capacity:
            shrink(self.fd, self.offset, self.dtype, (rows, self.info.dim))
        sync_file(self.fd)
        os.close(self.fd)
        self.fd = -1
        entries = {
            "vectors": (
                tensor_code(np.empty(0, self.dtype)),
                (rows, self.info.dim),
                self.digest.hexdigest(),
            )
        }
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
        write_manifest(
            self.tmp,
            entries,
            metadata,
            self.tmp / f".manifest.json.writing-{secrets.token_hex(4)}",
            sync_file,
        )
        publish_directory(self.tmp, self.store / shard_name(self.first_row))
        return rows

    def abort(self) -> None:
        """Drop the shard: its views must be gone; removes the temporary directory."""
        self._open.clear()
        self.release()
        for mapping in self._maps:
            with contextlib.suppress(BufferError):  # a view outlives the run: the OS frees it
                mapping.close()
        self._maps.clear()
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        shutil.rmtree(self.tmp, ignore_errors=True)


def _write(file: Path, array: ndarray) -> str:
    """A small tensor as a synced .npy file; returns the sha256 of its data."""
    data = np.ascontiguousarray(array)
    fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    with os.fdopen(fd, "wb") as f:
        npy_format.write_array_header_1_0(f, npy_format.header_data_from_array_1_0(data))
        f.write(data.tobytes())
        f.flush()
        sync_file(f.fileno())
    return hashlib.sha256(data.tobytes()).hexdigest()
