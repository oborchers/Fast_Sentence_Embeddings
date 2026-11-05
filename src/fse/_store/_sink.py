#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.ShardSink``: the rows of an encoder written straight into the shards of a store.

The worker of a task writes its rows into a buffer of the task (stage 5 and the inverse
norms in the worker); its commit writes them into the current shard at their offset. A
task that crosses the end of a shard is split over the shards at its commit. A shard
whose rows are all committed is published at once, then ``published_rows`` grows.
With full durability its successful sync also grows ``durable_rows``; none keeps it zero.
After an error the committed rows are published and the rest is dropped. The reservation,
the threads, the publication and the closing are those of every shard store
(``ShardSinkBase``); this module adds what a row of sentence vectors holds.
"""

from __future__ import annotations

import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray

from fse import _codes, _log
from fse._encoder import Encoder
from fse._ids import checked_ids, id_block
from fse._validation import PositiveInt, validate_arguments

from ._base_sink import ShardSinkBase
from ._layout import create_store, read_store
from ._recovery import refuse_unsynced, snapshot
from ._shard import ShardWriter
from ._vectors_format import StoreInfo

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


class ShardSink(ShardSinkBase):
    """Write the rows of one run of an encoder into the store at ``path``: a new store, or
    the end of an existing one of the same encoder, dtype and dimension.

    Use it as a context manager (or call ``close``): leaving it publishes the committed
    rows, also after an error. ``first_row`` is the row of the store the first row of the
    run becomes; ``published_rows`` counts visible leading rows. ``durable_rows`` counts
    rows published and synced with ``durability="full"`` and stays zero under "none".
    ``seal`` ends the current shard early, so ``publish`` can publish its rows without
    waiting for a full shard. Store creation always uses full durability.
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
        durability: Literal["full", "none"] = "full",
        doc_ids: bool = False,
    ):
        if not isinstance(encoder, Encoder):
            raise TypeError(f"encoder must be an Encoder, not {type(encoder).__name__}")
        path = Path(path)
        self._start = time.perf_counter()
        info = StoreInfo(
            encoder.fingerprint, dtype, encoder.output_dim, doc_ids, 2 if doc_ids else 1
        )
        new = not path.exists() and not path.is_symlink()
        if new:
            create_store(path, info)
        existing = read_store(path)
        existing.check(info, path)
        info = existing
        synced = durability == "full"
        checked = snapshot(path, info, verify=False)
        try:
            if synced:
                refuse_unsynced(path, checked.shards)
            first_row = sum(shard.rows for shard, _ in checked.shards)
        finally:
            checked.shards.clear()  # a sink keeps no mappings, also on refusal

        super().__init__(
            path,
            synced=synced,
            shard_rows=shard_rows or default_shard_rows(info.dim, dtype),
            first_row=first_row,
        )
        self.has_doc_ids = doc_ids
        self._info = info
        self.dim, self.dtype, self.encoder_fingerprint = info.dim, dtype, info.encoder
        self._created_by = _created_by()
        _log.emit(
            "store.sink_opened",
            "fse store: a sink %s a store at row %d: %d dimensions, %s of %d rows, durability=%s",
            "creates" if new else "appends to",
            self.first_row,
            info.dim,
            dtype,
            self.shard_rows,
            durability,
            mode="new" if new else "append",
            rows=self.first_row,
            dim=info.dim,
            dtype=dtype,
            shard_rows=self.shard_rows,
            durability=durability,
        )

    # --- what a row of sentence vectors holds ------------------------------------------

    def _new_writer(self, store_row: int) -> ShardWriter:
        return ShardWriter(
            self.path,
            self._info,
            store_row,
            self.shard_rows,
            self._created_by,
            synced=self._synced,
        )

    def _own(self, n_rows: int) -> tuple:
        codes = np.empty((n_rows, self.dim), _codes.STORAGE_TYPES[self.dtype])
        scales = np.empty(n_rows, np.float32) if self.dtype == "int8" else None
        return codes, scales, np.empty(n_rows, np.float32)

    def _flat(self, buffers: tuple) -> tuple[ndarray, ...]:
        """A target as the sink protocol hands it out: codes, [scales,] inverse norms."""
        codes, scales, norms = buffers
        return (codes, norms) if scales is None else (codes, scales, norms)

    def _commit_target(self, first_row, rows, doc_ids):
        parts, own = self._claim(first_row, rows)
        if self.has_doc_ids != (doc_ids is not None):
            raise ValueError("doc_ids must match the mode of the sink")
        ids = None
        if doc_ids is not None:
            checked_ids(doc_ids)
            if len(doc_ids) != sum(len(part[2][0]) for part in parts):
                raise ValueError("doc_ids must have one ID per committed row")
            ids = id_block(doc_ids, 0, len(doc_ids))
        del self._targets[first_row]  # only once the commit is checked
        return parts, own, ids

    def commit(self, first_row: int, rows: Any, *, doc_ids: Any = None) -> None:
        parts, own, ids = self._commit_target(first_row, rows, doc_ids)
        offset = 0
        for writer, begin, (codes, scales, norms) in parts:
            if own is not None:
                n = len(codes)
                codes[...] = own[0][offset : offset + n]
                if scales is not None:
                    scales[...] = own[1][offset : offset + n]
                norms[...] = own[2][offset : offset + n]
            # The write first: rows counted as committed always have their write queued,
            # also when an interrupt (Ctrl+C) stops this between the two hand-overs.
            self._submit_rows(
                writer, begin, codes, None if ids is None else ids[offset : offset + len(codes)]
            )
            offset += len(codes)
            self._handed_over(writer, len(codes))

    def _submit_rows(self, writer, begin, codes, ids):
        if ids is None:
            self._io.submit(writer.write, begin, codes, nbytes=codes.nbytes)
            self._hash.submit(writer.commit, begin, codes, nbytes=codes.nbytes)
        else:
            size = codes.nbytes + ids.nbytes
            self._io.submit(writer.write, begin, codes, ids, nbytes=size)
            self._hash.submit(writer.commit, begin, codes, ids, nbytes=size)

    # --- the log events ----------------------------------------------------------------

    def _closed_event(self, outcome: str, dropped: int) -> None:
        _log.emit(
            "store.sink_closed",
            "fse store: a sink closed (%s): %d rows published, %d durable in %d shards, %d dropped",
            outcome,
            self.published_rows,
            self.durable_rows,
            self._published,
            dropped,
            rows=self._committed,
            shards=self._published,
            published_rows=self.published_rows,
            durable_rows=self.durable_rows,
            dropped_rows=dropped,
            durability=self.durability,
            outcome=outcome,
            seconds=time.perf_counter() - self._start,
        )

    def _published_event(self, writer, rows: int) -> None:
        _log.emit(
            "store.shard_published",
            "fse store: published a shard of %d rows at row %d; %d rows published, %d durable",
            rows,
            writer.first_row,
            self.published_rows,
            self.durable_rows,
            first_row=writer.first_row,
            rows=rows,
            published_rows=self.published_rows,
            durable_rows=self.durable_rows,
        )
