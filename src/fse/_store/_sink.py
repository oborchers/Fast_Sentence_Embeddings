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
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from fse import _codes, _log
from fse._encoder import Encoder
from fse._ids import checked_ids, id_block
from fse._partition import partition
from fse._validation import (
    PositiveInt,
    UncheckedPath,
    file_path,
    validate_arguments,
)

from ._base_sink import ShardSinkBase
from ._bits_write import backfill as backfill_bits
from ._layout import check_root_size, create_store, read_store
from ._position_format import MODE as SOURCE_POSITION
from ._position_format import SourcePosition, anchor_of, check_position
from ._recovery import clear_exception_frames, refuse_unsynced, snapshot
from ._resume_format import MODE as SOURCE_ROWS
from ._resume_format import SourceRows, check_mode, check_source, sync_store
from ._shard import ShardWriter
from ._stream_state import StreamState
from ._vectors_format import VECTORS, StoreInfo

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


def stream_of(info: Any, shards: list) -> StreamState | None:
    """The stream state of a store of the mode ``source_position@1`` with the ``shards`` of its
    snapshot, else None."""
    anchor = anchor_of(info, shards)
    if anchor is None:
        return None
    chain = tuple((shard.first_row + shard.rows, shard.anchor) for shard, _ in shards)
    return StreamState(info.resume, anchor, sum(shard.rows for shard, _ in shards), chain)


def _open_store(
    path: Path, info: StoreInfo, *, new: bool, resume: bool, synced: bool
) -> tuple[StoreInfo, int, StreamState | None]:
    """The info of the store ``path`` (made if it is new, unless it resumes: it is made once
    a run brings its source), the rows it holds and, for a store of the mode
    ``source_position@1``, its stream state."""
    if new and resume:
        return info, 0, None
    if new:
        create_store(path, info)
    existing = read_store(path, VECTORS.name)
    check_mode(existing, resume, path)
    existing.check(info, path)
    checked = snapshot(path, existing, verify=False)
    try:
        if synced:
            refuse_unsynced(path, checked.shards)
        rows = sum(shard.rows for shard, _ in checked.shards)
        return existing, rows, stream_of(existing, checked.shards)
    finally:
        checked.shards.clear()  # a sink keeps no mappings, also on refusal


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

    ``dtype`` is the stored dtype of the rows (``"float32"``, ``"float16"`` or ``"int8"``).
    ``shard_rows`` is the rows of a shard (default: the largest multiple of 16,384 rows
    within 256 MiB of vectors, at most 2**20; fewer rows for very wide vectors). ``doc_ids=True`` makes a store with a
    document ID per row; ``Encoder.encode(..., doc_ids=ids)`` then requires the IDs, and
    appending to a store needs the same ID mode. ``resume=True`` makes the store
    resumable (``Encoder.encode`` with the same source, or ``encode_batches`` with
    ``positions=True``); the sink then takes rows only from those calls. ``sign_bits=True``
    also writes the sign bits of each shard for ``search(mode="binary")``.
    """

    inverse_norms = True
    _NAME = "ShardSink"

    @validate_arguments
    def __init__(
        self,
        path: UncheckedPath,
        encoder: Any,
        *,
        dtype: Literal["float32", "float16", "int8"] = "float32",
        shard_rows: PositiveInt | None = None,
        durability: Literal["full", "none"] = "full",
        doc_ids: bool = False,
        resume: bool = False,
        sign_bits: bool = False,
    ):
        if not isinstance(encoder, Encoder):
            raise TypeError(f"encoder must be an Encoder, not {type(encoder).__name__}")
        path = Path(file_path(path, "path"))
        self._start = time.perf_counter()
        info = StoreInfo(
            encoder.fingerprint, dtype, encoder.output_dim, doc_ids, 2 if doc_ids or resume else 1
        )
        new = not path.exists() and not path.is_symlink()
        synced = durability == "full"
        info, first_row, stream = _open_store(path, info, new=new, resume=resume, synced=synced)

        created_by = _created_by()  # before the sink is initialized
        super().__init__(
            path,
            synced=synced,
            shard_rows=shard_rows or default_shard_rows(info.dim, dtype),
            first_row=first_row,
            row_bytes=(
                info.dim * np.dtype(_codes.STORAGE_TYPES[dtype]).itemsize
                + (4 if dtype == "int8" else 0)
                + 4
                + (8 if doc_ids else 0)
            ),
        )
        self.has_doc_ids = doc_ids
        self._info = info
        self._stream = stream
        self.dim, self.dtype, self.encoder_fingerprint = info.dim, dtype, info.encoder
        self._created_by = created_by
        self.resume = resume
        self.sign_bits = sign_bits
        self._new, self._kept = new, 0
        if resume:  # the store is made or compared once a run brings its source
            self._ready = False
            return
        self._opened_event(new, durability)

    def target(self, first_row: int, n_rows: int) -> tuple[NDArray[Any], ...]:
        """Reserve the stored vector arrays for one task."""
        return self._target(first_row, n_rows)

    def _opened_event(self, new: bool, durability: str, **resumed: Any) -> None:
        info = self._info
        _log.emit(
            "store.sink_opened",
            "fse store: a sink %s a store at row %d: %d dimensions, %s of %d rows, durability=%s",
            "creates" if new else "appends to",
            self.first_row,
            info.dim,
            self.dtype,
            self.shard_rows,
            durability,
            mode="new" if new else "append",
            rows=self.first_row,
            dim=info.dim,
            dtype=self.dtype,
            shard_rows=self.shard_rows,
            durability=durability,
            **({"sign_bits": True} if self.sign_bits else {}),
            **resumed,
        )

    # --- resuming ----------------------------------------------------------------------

    def _attach_source(self, identity: Any, shard: tuple[int, int], seconds: float) -> int:
        """Bind this sink to the source of a run (``Encoder.encode(texts, shard=)``): make the
        store with its identity, or compare the identity with that of the store, which
        raises a ValueError before any row is reserved. Verifies the published shards,
        makes the entries of the store durable under full durability and returns the rows
        the store holds (the rows of the part that are done)."""
        self._check_open()
        if self._ready:
            raise RuntimeError("the sink took a source already; make a new sink for another run")
        first, _ = partition(identity.rows, *shard)
        wanted = SourceRows(identity.fingerprint, identity.form, identity.rows, shard, first)
        if self._new:
            create_store(self.path, replace(self._info, resume=wanted))
        info = read_store(self.path, VECTORS.name)
        check_mode(info, True, self.path, SOURCE_ROWS)
        info.check(self._info, self.path)
        check_source(info, wanted, self.path)
        checked = snapshot(self.path, info, verify=True)
        try:
            if self._synced:
                refuse_unsynced(self.path, checked.shards)
            rows = sum(shard_.rows for shard_, _ in checked.shards)
            self._backfill(info, checked.shards)
        finally:
            checked.shards.clear()
        if self._synced:
            sync_store(self.path)
        self._info, self.first_row, self._kept, self._ready = info, rows, rows, True
        self._opened_event(
            self._new,
            self.durability,
            resume=SOURCE_ROWS,
            kept_rows=rows,
            source_rows=identity.rows,
            source_bytes=identity.nbytes,
            source_seconds=seconds,
            source_fingerprint=identity.fingerprint,
        )
        return rows

    def _attach_stream(self, position: str | None) -> int:
        """Bind this sink to a stream (``Encoder.encode_batches(batches, sink=, positions=True)``).
        An existing store is compared with the sink, verified and made durable under full
        durability, and its rows are returned. A new store is made now, with the anchor
        ``(position, 0)`` of the first batch (the size of its root is checked before it is
        made). A second attach is a RuntimeError."""
        self._check_open()
        if self._ready:
            raise RuntimeError("the sink took a source already; make a new sink for another run")
        if self._new:
            info = replace(self._info, resume=SourcePosition(check_position(position), 0))
            check_root_size(info)
            create_store(self.path, info)
        info = read_store(self.path, VECTORS.name)
        check_mode(info, True, self.path, SOURCE_POSITION)
        info.check(self._info, self.path)
        checked = snapshot(self.path, info, verify=True)
        try:
            if self._synced:
                refuse_unsynced(self.path, checked.shards)
            rows = sum(shard_.rows for shard_, _ in checked.shards)
            stream = stream_of(info, checked.shards)
            self._backfill(info, checked.shards)
        finally:
            checked.shards.clear()
        if self._synced:
            sync_store(self.path)
        assert stream is not None
        self._info, self.first_row, self._kept, self._ready = info, rows, rows, True
        self._stream = stream
        self._opened_event(self._new, self.durability, resume=SOURCE_POSITION, kept_rows=rows)
        return rows

    def _backfill(self, info: StoreInfo, shards: list) -> None:
        """With ``sign_bits=True``, write the bits that the published shards of a resumed store
        lack (a run that stopped between a shard and its bits): the shards are checked, so
        the bits are those the uninterrupted run would have written. An append sink does not
        read the whole store, so it leaves them to ``fse.add_sign_bits``."""
        if self.sign_bits:
            start = time.perf_counter()
            try:
                backfill_bits(self.path, info, shards, self._synced, 1, start)
            except BaseException as error:
                clear_exception_frames(error)  # the frames hold the mappings of the shards
                raise

    def _check_stream(self) -> None:
        """The checks of ``encode_batches(positions=True)`` that read nothing: the sink resumes,
        took no run yet, and an existing store is of the mode ``source_position@1``."""
        self._check_open()
        if self._ready:
            raise RuntimeError("the sink took a source already; make a new sink for another run")
        if not self._new:
            check_mode(self._info, True, self.path, SOURCE_POSITION)

    # --- what a row of sentence vectors holds ------------------------------------------

    def _new_writer(self, store_row: int) -> ShardWriter:
        return ShardWriter(
            self.path,
            self._info,
            store_row,
            self.shard_rows,
            self._created_by,
            synced=self._synced,
            sign_bits=self.sign_bits,
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
            **({"skipped_rows": self._kept} if self.resume else {}),
            **({"sign_bits_shards": self._extras_published} if self.sign_bits else {}),
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
