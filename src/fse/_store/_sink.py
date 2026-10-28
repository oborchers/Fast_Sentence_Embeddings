#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""``fse.ShardSink``: the rows of an encoder written straight into the shards of a store.

The worker of a task writes its rows into a mapped range of the current shard (stage 5
and the inverse norms in the worker); a task that crosses the end of a shard gets arrays
of its own, copied into the shards at its commit. A shard whose rows are all committed is
published at the next call of the sink once no view of its mappings is left (at the
latest at ``flush``), and only then ``durable_rows`` grows: a stream confirms its input up
to there. After an error the committed rows are published and the rest is dropped.
"""

from __future__ import annotations

import traceback
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._encoder import Encoder
from fse._validation import PositiveInt, validate_arguments

from ._layout import StoreInfo, create_store, read_store, shards
from ._shard import ShardWriter

# Shards hold whole multiples of this many rows by default, and at most this many bytes
# of vectors or rows.
_ALIGN_ROWS = 16_384
_SHARD_BYTES = 256 * 2**20
_MAX_SHARD_ROWS = 2**20


def default_shard_rows(dim: int, dtype: str) -> int:
    """The largest multiple of 16,384 rows within 256 MiB of vectors (at most 2**20), or
    the rows that fit for very wide vectors."""
    row_bytes = dim * np.dtype(_codes.STORAGE_TYPES[dtype]).itemsize
    rows = min(_MAX_SHARD_ROWS, _SHARD_BYTES // row_bytes)
    return max(1, rows - rows % _ALIGN_ROWS if rows >= _ALIGN_ROWS else rows)


def _created_by() -> str:
    try:
        return f"fse {version('fse')}"
    except PackageNotFoundError:  # pragma: no cover
        return "fse"


class ShardSink:
    """Write the rows of one run of an encoder into the store at ``path``: a new store, or
    the end of an existing one of the same encoder, dtype and dimension.

    Use it as a context manager (or call ``close``): leaving it publishes the committed
    rows, also after an error. ``first_row`` is the row of the store the first row of the
    run becomes; ``durable_rows`` counts the leading rows of the run that are published
    and synced; ``seal`` ends the current shard early, so ``publish`` can make its rows
    durable without waiting for a full shard.
    """

    inverse_norms = True

    @validate_arguments
    def __init__(
        self,
        path: Any,
        encoder: Any,
        *,
        dtype: Literal["float32", "float16", "int8"] = "float32",
        shard_rows: PositiveInt | None = None,
    ):
        if not isinstance(encoder, Encoder):
            raise TypeError(f"encoder must be an Encoder, not {type(encoder).__name__}")
        self.path = Path(path)
        info = StoreInfo(encoder.fingerprint, dtype, encoder.output_dim)
        if not self.path.exists() and not self.path.is_symlink():
            create_store(self.path, info)
        read_store(self.path).check(info, self.path)
        self.first_row = sum(shard.rows for shard, _ in shards(self.path, info))
        self.info = info
        self.dim, self.dtype, self.encoder = info.dim, dtype, info.encoder
        self.shard_rows = shard_rows or default_shard_rows(info.dim, dtype)
        self.reserved_rows = self.committed_rows = self.durable_rows = 0
        self._writers: deque[ShardWriter] = deque()
        self._targets: dict[int, tuple[list, tuple | None]] = {}  # run row -> parts, own
        self._closed = False
        self._created_by = _created_by()

    # --- the sink protocol, in the calling thread -------------------------------------

    def target(self, first_row: int, n_rows: int) -> tuple[ndarray, ...]:
        self._check_open()
        if first_row != self.reserved_rows:
            raise ValueError(
                f"rows from {first_row} reserved, expected rows from {self.reserved_rows}"
            )
        self._publish_ready()
        parts = []
        left = n_rows
        while left:
            writer = self._writable()
            take = min(left, writer.end - writer.reserved)
            parts.append((writer, writer.reserved, writer.reserve(take)))
            self.reserved_rows += take
            left -= take
        if len(parts) == 1:
            self._targets[first_row] = (parts, None)
            return _flat(parts[0][2])
        own = self._own(n_rows)
        self._targets[first_row] = (parts, own)
        return _flat(own)

    def commit(self, first_row: int, rows: Any) -> None:
        self._check_open()
        if first_row != self.committed_rows or first_row not in self._targets:
            raise ValueError(
                f"rows from {first_row} committed, expected rows from {self.committed_rows}"
            )
        parts, own = self._targets[first_row]
        given = rows if isinstance(rows, tuple) else (rows,)
        expected = _flat(own if own is not None else parts[0][2])
        if len(given) != len(expected) or not all(
            _same(g, e) for g, e in zip(given, expected, strict=True)
        ):
            raise ValueError(f"the rows from {first_row} are not the target reserved for them")
        del self._targets[first_row]  # only once the commit is checked
        offset = 0
        for writer, begin, (codes, scales, norms) in parts:
            if own is not None:
                n = len(codes)
                codes[...] = own[0][offset : offset + n]
                if scales is not None:
                    scales[...] = own[1][offset : offset + n]
                norms[...] = own[2][offset : offset + n]
                offset += n
            writer.commit(begin, codes)
            self.committed_rows += len(codes)
        del parts, own, given, expected, codes, scales, norms
        self._publish_ready()

    # --- durability --------------------------------------------------------------------

    def seal(self) -> None:
        """End the current shard at the rows reserved so far; it is published once they
        are committed and no view of them is left."""
        self._check_open()
        if self._writers and not self._writers[-1].full:
            writer = self._writers[-1]
            if writer.reserved == 0:
                self._writers.pop().abort()
            else:
                writer.seal()

    def publish(self) -> int:
        """Publish every shard whose rows are all committed; returns ``durable_rows``."""
        self._check_open()
        self._publish_ready()
        return self.durable_rows

    def flush(self) -> int:
        """Seal and publish everything committed; every reserved row must be committed.
        Returns ``durable_rows``."""
        self._check_open()
        if self.reserved_rows != self.committed_rows:
            raise RuntimeError(
                f"{self.reserved_rows - self.committed_rows} reserved rows were not committed"
            )
        self.seal()
        self._publish_ready()
        if self._writers:
            raise RuntimeError(
                "a view of the rows of a shard is still in use; nothing more was published"
            )
        return self.durable_rows

    def close(self) -> None:
        """Publish the committed rows and drop the rest; the sink takes no more rows."""
        if self._closed:
            return
        try:
            self._abort_uncommitted()
        finally:
            for writer in self._writers:
                writer.abort()
            self._writers.clear()
            self._closed = True

    def __enter__(self) -> ShardSink:
        return self

    def __exit__(self, *exc: object) -> None:
        tb = exc[2]
        if isinstance(tb, TracebackType):
            traceback.clear_frames(tb)  # the frames of the run may hold views of the shards
        self.close()

    # --- internals ---------------------------------------------------------------------

    def _check_open(self):
        if self._closed:
            raise RuntimeError("the ShardSink is closed; make a new one for another run")

    def _writable(self) -> ShardWriter:
        if not self._writers or self._writers[-1].full:
            store_row = self.first_row + self.reserved_rows
            self._writers.append(
                ShardWriter(self.path, self.info, store_row, self.shard_rows, self._created_by)
            )
        return self._writers[-1]

    def _own(self, n_rows: int) -> tuple:
        codes = np.empty((n_rows, self.dim), _codes.STORAGE_TYPES[self.dtype])
        scales = np.empty(n_rows, np.float32) if self.dtype == "int8" else None
        return codes, scales, np.empty(n_rows, np.float32)

    def _publish_ready(self) -> None:
        while self._writers and self._writers[0].can_finish():
            writer = self._writers[0]
            try:
                rows = writer.finish()
            except FileExistsError:
                self._writers.popleft().abort()
                raise FileExistsError(
                    f"another writer appended to {self.path} at row {writer.first_row} first; "
                    f"this run published its first {self.durable_rows} rows (rows "
                    f"{self.first_row} to {self.first_row + self.durable_rows - 1} of the store)"
                ) from None
            self._writers.popleft()
            self.durable_rows += rows

    def _abort_uncommitted(self) -> None:
        """Publish the committed rows of every shard, end it there, drop the rest."""
        self._targets.clear()  # uncommitted targets are never committed
        kept: deque[ShardWriter] = deque()
        for writer in self._writers:
            writer.drop_uncommitted()
            if writer.committed:
                kept.append(writer)
            else:
                writer.abort()
        self._writers = kept
        self.reserved_rows = self.committed_rows
        self._publish_ready()
        if self._writers:
            raise RuntimeError(
                "a view of the rows of a shard is still in use; its committed rows were not "
                f"published (durable_rows {self.durable_rows})"
            )


def _flat(parts: tuple) -> tuple[ndarray, ...]:
    """A target as the sink protocol hands it out: codes, [scales,] inverse norms."""
    codes, scales, norms = parts
    return (codes, norms) if scales is None else (codes, scales, norms)


def _same(given: Any, reserved: ndarray) -> bool:
    return (
        isinstance(given, ndarray)
        and given.shape == reserved.shape
        and given.__array_interface__["data"][0] == reserved.__array_interface__["data"][0]
    )
