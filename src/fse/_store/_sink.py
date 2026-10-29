#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""``fse.ShardSink``: the rows of an encoder written straight into the shards of a store.

The worker of a task writes its rows into a buffer of the task (stage 5 and the inverse
norms in the worker); its commit writes them into the current shard at their offset. A
task that crosses the end of a shard is split over the shards at its commit. A shard
whose rows are all committed is published at once, and only then ``durable_rows`` grows:
a stream confirms its input up to there. After an error the committed rows are published
and the rest is dropped.
"""

from __future__ import annotations

from collections import deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._encoder import Encoder
from fse._validation import PositiveInt, validate_arguments

from ._layout import StoreInfo, create_store, read_store, shards
from ._shard import ShardWriter
from ._writer import WriterThread

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
        self.dim, self.dtype, self.encoder_fingerprint = info.dim, dtype, info.encoder
        self.shard_rows = shard_rows or default_shard_rows(info.dim, dtype)
        self.reserved_rows = self.committed_rows = self.durable_rows = 0
        self._writers: deque[ShardWriter] = deque()  # shards still taking rows
        self._made: set[ShardWriter] = set()  # shards of the run not yet published
        self._targets: dict[int, tuple[list, tuple | None]] = {}  # run row -> parts, own
        self._closed = False
        self._created_by = _created_by()
        # Hashing (in order), writing and publishing (syncs, a rename) in three threads, so
        # the flush of one shard does not stop the hashing and writing of the next.
        self._hash = WriterThread("fse-shard-hash")
        self._io = WriterThread("fse-shard-write")
        self._publisher = WriterThread("fse-shard-publish")

    # --- the sink protocol, in the calling thread -------------------------------------

    def target(self, first_row: int, n_rows: int) -> tuple[ndarray, ...]:
        self._check_open()
        self._check_threads()  # a failed write first: no row after it is taken
        if first_row != self.reserved_rows:
            raise ValueError(
                f"rows from {first_row} reserved, expected rows from {self.reserved_rows}"
            )
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
        self._check_threads()
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
            # The write first: rows counted as committed always have their write queued,
            # also when an interrupt (Ctrl+C) stops this between the two hand-overs.
            self._io.submit(writer.write, begin, codes, nbytes=codes.nbytes)
            self._hash.submit(writer.commit, begin, codes, nbytes=codes.nbytes)
            writer.queued += len(codes)
            self.committed_rows += len(codes)
            self._publish_ready()  # a full shard goes on at once, so few files stay open

    # --- durability --------------------------------------------------------------------

    def seal(self) -> None:
        """End the current shard at the rows reserved so far; it is published once they
        are committed."""
        self._check_open()
        if self._writers and not self._writers[-1].full:
            writer = self._writers[-1]
            if writer.reserved == 0:
                self._writers.pop().abort()
            else:
                writer.seal()

    def publish(self) -> int:
        """Publish every shard whose rows are all committed and wait for it; returns
        ``durable_rows``."""
        self._check_open()
        self._publish_ready()
        self._drain()
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
        self._drain()
        return self.durable_rows

    def close(self) -> None:
        """Publish the committed rows and drop the rest; the sink takes no more rows."""
        if self._closed:
            return
        try:
            self._settle()  # the rows handed over are written, unless a thread failed
            if self._failed():  # nothing more is published
                self._raise_unreported()
            else:
                self._abort_uncommitted()
                self._drain()
        finally:
            self._hash.stop()  # first: it hands finished shards to the publisher
            self._io.stop()
            self._publisher.stop()
            for writer in list(self._made):  # what was not published
                writer.abort()
            self._writers.clear()
            self._made.clear()
            self._closed = True

    def __enter__(self) -> ShardSink:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- internals ---------------------------------------------------------------------

    def _check_threads(self) -> None:
        for thread in (self._hash, self._io, self._publisher):
            thread.check()

    def _drain(self) -> None:
        for thread in (self._hash, self._io, self._publisher):  # hash hands on to publisher
            thread.drain()

    def _settle(self) -> None:
        for thread in (self._hash, self._io, self._publisher):
            thread.settle()

    def _raise_unreported(self) -> None:
        """Raise the cause of a failure no call has raised yet: the writing thread first,
        since the others wait for it and fail after it."""
        threads = (self._io, self._hash, self._publisher)
        if not any(t.reported for t in threads):
            for thread in threads:
                if thread.failed:
                    thread.check()

    def _failed(self) -> bool:
        return any(t.failed for t in (self._hash, self._io, self._publisher))

    def _check_open(self):
        if self._closed:
            raise RuntimeError("the ShardSink is closed; make a new one for another run")

    def _writable(self) -> ShardWriter:
        if not self._writers or self._writers[-1].full:
            store_row = self.first_row + self.reserved_rows
            writer = ShardWriter(self.path, self.info, store_row, self.shard_rows, self._created_by)
            self._writers.append(writer)
            self._made.add(writer)
        return self._writers[-1]

    def _own(self, n_rows: int) -> tuple:
        codes = np.empty((n_rows, self.dim), _codes.STORAGE_TYPES[self.dtype])
        scales = np.empty(n_rows, np.float32) if self.dtype == "int8" else None
        return codes, scales, np.empty(n_rows, np.float32)

    def _publish_ready(self) -> None:
        """Hand every shard whose rows are all handed over to be published, in order."""
        while self._writers and self._writers[0].queued_all:
            # After its last hash, in order, the shard goes to the publisher.
            self._hash.submit(self._hand_on, self._writers.popleft())

    def _hand_on(self, writer: ShardWriter) -> None:
        """Hand ``writer`` to the publisher (in the hash thread): an error of the publisher
        met here is the publisher's, not yet reported to the caller."""
        self._publisher.submit(self._finish, writer, report=False)

    def _finish(self, writer: ShardWriter) -> None:
        """Publish ``writer`` (in the writer thread, after its commits)."""
        try:
            rows = writer.finish(lambda: self._io.failed)
        except FileExistsError:
            writer.abort()
            raise FileExistsError(
                f"another writer appended to {self.path} at row {writer.first_row} first; "
                f"this run published its first {self.durable_rows} rows (rows "
                f"{self.first_row} to {self.first_row + self.durable_rows - 1} of the store)"
            ) from None
        except BaseException:
            writer.abort()
            raise
        self._made.discard(writer)
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
