#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.ScoreSink``: the scores of heads, a group or a single head, written straight into
the shards of a score store (docs/architecture/scores.md, "Writing").

The sink is a float32 sink of ``dim`` values per row, the columns of all heads side by side.
``predict_batches(score_sink=sink)`` of ``fse.Heads``, ``Classifier`` or ``Regressor`` fills
them: the adapter of the run copies the output of each head into its columns. The constructor checks everything but the path of
each head and creates nothing for a new store; ``predict_batches`` attaches the run,
before it reads a text, and passes the paths, and only then is ``store.json`` written, whole
and for good. The reservation, the threads, the publication and the closing are those of
every shard store (``ShardSinkBase``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from fse import _log
from fse._ids import checked_ids, id_block
from fse._partition import partition
from fse._store._base_sink import ShardSinkBase
from fse._store._layout import check_root_size, create_store, read_store
from fse._store._position_format import MODE as SOURCE_POSITION
from fse._store._position_format import SourcePosition, check_position
from fse._store._recovery import refuse_unsynced, snapshot
from fse._store._resume_format import MODE as SOURCE_ROWS
from fse._store._resume_format import SourceRows, check_mode, check_source, sync_store
from fse._store._score_shard import ScoreShardWriter
from fse._store._scores_format import SCORES
from fse._store._sink import default_shard_rows, stream_of
from fse._validation import (
    PositiveInt,
    UncheckedPath,
    file_path,
    validate_arguments,
)

from ._schema import ScoreSchema


def _created_by() -> str:
    try:
        return f"fse {version('fse')}"
    except PackageNotFoundError:  # pragma: no cover
        return "fse"


class ScoreSink(ShardSinkBase):
    """Write the scores of heads into the score store at ``path``: a new store, or the end
    of an existing one of the same heads. ``heads`` is a ``fse.Heads``, or one ``Classifier``
    or ``Regressor`` with its ``name`` (required: the name is written to the store, and
    ``ScoreStore`` reads the head by it). What a store holds is read through
    ``ScoreStore.heads``; the sink gives out no schema.

    ``stored`` says what each head stores: None (the decision of a classifier, the values of
    a regressor), one name for all heads, or a mapping of head names to ``decision``,
    ``proba`` or ``values``. Give the sink to ``predict_batches(score_sink=...)`` of the same
    heads or head; it takes rows from nothing else. A new store is created when that call attaches the run,
    not before; closing a sink that never attached leaves nothing behind.

    Use it as a context manager (or call ``close``): leaving it publishes the committed rows,
    also after an error. ``first_row`` is the row of the store the first row of the run
    becomes; ``published_rows`` counts visible leading rows; ``durable_rows`` counts rows
    published and synced with ``durability="full"`` and stays zero under "none".

    With ``resume=True`` the sink resumes: give it to ``predict(texts, score_sink=sink,
    shard=)`` of the same heads, which reads a source that can be read again, compares its
    identity with the store and runs the rows the store does not hold yet
    (docs/architecture/scores.md, "Resuming"). It takes rows from nothing else.
    """

    inverse_norms = False
    dtype = "float32"
    _NAME = "ScoreSink"

    @validate_arguments
    def __init__(
        self,
        path: UncheckedPath,
        heads: Any,
        *,
        name: Any = None,
        stored: Any = None,
        doc_ids: bool = False,
        shard_rows: PositiveInt | None = None,
        durability: Literal["full", "none"] = "full",
        resume: bool = False,
    ):
        schema = ScoreSchema.from_heads(heads, stored, name)
        path = Path(file_path(path, "path"))
        self._start = time.perf_counter()
        self._exists = path.exists() or path.is_symlink()
        synced = durability == "full"
        first_row, stream = 0, None
        if self._exists:
            first_row, stream = _existing(path, schema, doc_ids, synced, resume)
        created_by = _created_by()  # before the sink is initialized
        row_values = schema.dim + (2 if doc_ids else 0)  # the 8 bytes of an ID, as float32 values
        super().__init__(
            path,
            synced=synced,
            shard_rows=shard_rows or default_shard_rows(row_values, "float32"),
            first_row=first_row,
        )
        self.has_doc_ids = doc_ids
        self.dim, self.encoder_fingerprint = schema.dim, schema.encoder
        self._schema, self._info = schema, None
        self._stream = stream
        self._created_by = created_by
        self._attached = False
        self.resume = resume
        self._kept = 0
        if resume:  # the store is made or compared once a run brings its source
            self._ready = False

    @property
    def _score_schema(self) -> ScoreSchema:
        """The schema of the scores the sink writes (the paths of the heads are filled in
        once a run attached)."""
        return self._schema

    @property
    def columns(self) -> dict[str, slice]:
        """The columns of each head in a row, by head name."""
        return self._schema.columns

    # --- the run attaches ---------------------------------------------------------------

    def _attach(self, paths: Mapping[str, str]) -> None:
        """Attach the run that ``predict_batches`` starts, with the path of each head:
        a new store gets its ``store.json`` now, whole; an existing one must hold the same
        paths. Called before the first text is read."""
        self._check_open()
        if self.resume:
            raise ValueError(
                "a ScoreSink made with resume=True takes the rows of predict(texts, "
                "score_sink=sink) of a source that can be read again; use a sink without "
                "resume for predict_batches"
            )
        if self._attached:
            raise RuntimeError("a ScoreSink takes one run; make a new one for another run")
        schema = self._schema.with_paths(paths)
        info = schema.info(self.has_doc_ids)
        created = False if self._exists else create_store(self.path, info)
        existing = read_store(self.path, SCORES.name)
        existing.check(info, self.path)
        if not self._exists:  # a store made now (or by a writer that was faster)
            self.first_row = _existing(self.path, schema, self.has_doc_ids, self._synced, False)[0]
        self._schema, self._info, self._attached = schema, existing, True
        self._opened_event(created)

    def _attach_source(
        self, identity: Any, shard: tuple[int, int], seconds: float, paths: Mapping[str, str]
    ) -> int:
        """Bind this sink to the source of a run (``predict(texts, score_sink=, shard=)``)
        with the path of each head: make the store with its identity, or compare the identity
        with that of the store, which raises a ValueError before any row is reserved.
        Verifies the published shards, makes the entries of the store durable under full
        durability and returns the rows the store holds (the rows of the part that are
        done)."""
        self._check_open()
        if self._ready:
            raise RuntimeError("the sink took a source already; make a new sink for another run")
        first, _ = partition(identity.rows, *shard)
        wanted = SourceRows(identity.fingerprint, identity.form, identity.rows, shard, first)
        schema = self._schema.with_paths(paths)
        info = replace(schema.info(self.has_doc_ids), resume=wanted)
        schema.check_info_size(info)  # before a store is made
        created = False if self._exists else create_store(self.path, info)
        existing = read_store(self.path, SCORES.name)
        check_mode(existing, True, self.path, SOURCE_ROWS)
        existing.check(info, self.path)
        check_source(existing, wanted, self.path)
        checked = snapshot(self.path, existing, verify=True)
        try:
            if self._synced:
                refuse_unsynced(self.path, checked.shards)
            _same_thresholds(schema, checked.shards)
            rows = sum(shard_.rows for shard_, _ in checked.shards)
        finally:
            checked.shards.clear()
        if self._synced:
            sync_store(self.path)
        self._schema, self._info = schema, existing
        self.first_row, self._kept, self._attached, self._ready = rows, rows, True, True
        self._opened_event(
            created,
            resume=SOURCE_ROWS,
            kept_rows=rows,
            source_rows=identity.rows,
            source_bytes=identity.nbytes,
            source_seconds=seconds,
            source_fingerprint=identity.fingerprint,
        )
        return rows

    def _attach_stream(self, position: str | None, paths: Mapping[str, str]) -> int:
        """Bind this sink to a stream (``predict_batches(..., score_sink=sink, positions=True)``)
        with the path of each head: an existing store is compared with the sink, verified and
        made durable under full durability, and its rows are returned. A new store is made now,
        with the anchor ``(position, 0)`` of the first batch (the size of its root is checked
        before it is made). A second attach is a RuntimeError."""
        self._check_open()
        if self._ready:
            raise RuntimeError("the sink took a source already; make a new sink for another run")
        schema = self._schema.with_paths(paths)
        info = schema.info(self.has_doc_ids)
        if not self._exists:
            info = replace(info, resume=SourcePosition(check_position(position), 0))
            check_root_size(info)
            create_store(self.path, info)
        existing = read_store(self.path, SCORES.name)
        check_mode(existing, True, self.path, SOURCE_POSITION)
        existing.check(info, self.path)
        checked = snapshot(self.path, existing, verify=True)
        try:
            if self._synced:
                refuse_unsynced(self.path, checked.shards)
            _same_thresholds(schema, checked.shards)
            rows = sum(shard_.rows for shard_, _ in checked.shards)
            stream = stream_of(existing, checked.shards)
        finally:
            checked.shards.clear()
        if self._synced:
            sync_store(self.path)
        assert stream is not None
        self._schema, self._info, self._stream = schema, existing, stream
        self.first_row, self._kept, self._attached, self._ready = rows, rows, True, True
        self._opened_event(not self._exists, resume=SOURCE_POSITION, kept_rows=rows)
        return rows

    def _check_stream(self) -> None:
        """The checks of ``predict_batches(positions=True)`` that read nothing: the sink
        resumes, took no run yet, and an existing store is of the mode ``source_position@1``."""
        self._check_open()
        if self._ready:
            raise RuntimeError("the sink took a source already; make a new sink for another run")
        self._schema.check_resume_size()  # the heads and the largest position must fit the root
        if self._exists:
            check_mode(read_store(self.path, SCORES.name), True, self.path, SOURCE_POSITION)

    def _opened_event(self, created: bool, **resumed: Any) -> None:
        schema = self._schema
        _log.emit(
            "scores.sink_opened",
            "fse scores: a sink %s a store at row %d: %d heads, %d columns, rows of %d, durability=%s",
            "creates" if created else "appends to",
            self.first_row,
            len(schema.heads),
            schema.dim,
            self.shard_rows,
            self.durability,
            mode="new" if created else "append",
            rows=self.first_row,
            heads=len(schema.heads),
            columns=schema.dim,
            shard_rows=self.shard_rows,
            durability=self.durability,
            doc_ids=self.has_doc_ids,
            **resumed,
        )

    # --- the sink protocol --------------------------------------------------------------

    def target(self, first_row: int, n_rows: int) -> NDArray[np.float32]:
        if not self._attached:
            raise RuntimeError(
                "a ScoreSink takes rows only from predict_batches(score_sink=...) of a "
                "Classifier, a Regressor or fse.Heads"
            )
        return self._target(first_row, n_rows)[0]

    def commit(self, first_row: int, rows: Any, *, doc_ids: Any = None) -> None:
        parts, own, ids = self._commit_target(first_row, rows, doc_ids)
        offset = 0
        for writer, begin, (matrix,) in parts:
            n = len(matrix)
            if own is not None:
                matrix[...] = own[0][offset : offset + n]
            piece = None if ids is None else ids[offset : offset + n]
            # The write first: rows counted as committed always have their write queued,
            # also when an interrupt (Ctrl+C) stops this between the two hand-overs.
            self._submit_rows(writer, begin, matrix, piece)
            offset += n
            self._handed_over(writer, n)

    def _commit_target(self, first_row: int, rows: Any, doc_ids: Any):
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

    def _submit_rows(self, writer: ScoreShardWriter, begin: int, matrix: ndarray, ids: Any) -> None:
        if ids is None:
            self._io.submit(writer.write, begin, matrix, nbytes=matrix.nbytes)
            self._hash.submit(writer.commit, begin, matrix, nbytes=matrix.nbytes)
        else:
            size = matrix.nbytes + ids.nbytes
            self._io.submit(writer.write, begin, matrix, ids, nbytes=size)
            self._hash.submit(writer.commit, begin, matrix, ids, nbytes=size)

    # --- what a row of scores holds -----------------------------------------------------

    def _new_writer(self, store_row: int) -> ScoreShardWriter:
        assert self._info is not None
        return ScoreShardWriter(
            self.path,
            self._info,
            self._schema.thresholds,
            store_row,
            self.shard_rows,
            self._created_by,
            synced=self._synced,
        )

    def _own(self, n_rows: int) -> tuple:
        return (np.empty((n_rows, self.dim), np.float32),)

    def _flat(self, buffers: tuple) -> tuple[ndarray, ...]:
        return buffers

    # --- the log events -----------------------------------------------------------------

    def _closed_event(self, outcome: str, dropped: int) -> None:
        _log.emit(
            "scores.sink_closed",
            "fse scores: a sink closed (%s): %d rows published, %d durable in %d shards, %d dropped",
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
        )

    def _published_event(self, writer: Any, rows: int) -> None:
        _log.emit(
            "scores.shard_published",
            "fse scores: published a shard of %d rows at row %d; %d rows published, %d durable",
            rows,
            writer.first_row,
            self.published_rows,
            self.durable_rows,
            first_row=writer.first_row,
            rows=rows,
            published_rows=self.published_rows,
            durable_rows=self.durable_rows,
        )


def _existing(
    path: Path, schema: ScoreSchema, doc_ids: bool, synced: bool, resume: bool
) -> tuple[int, Any]:
    """The rows of the existing store ``path``, which must hold the heads of ``schema`` (but
    for their paths, known later), and whose last shard must hold the thresholds of the
    heads: a run with other thresholds would break the chain of the shards. A resumable
    store is opened with ``resume=True`` only, and ``resume=True`` opens such a store only."""
    info = read_store(path, SCORES.name)
    check_mode(info, resume, path)
    info.check(schema.info(doc_ids), path, paths=False)
    checked = snapshot(path, info, verify=False)
    try:
        if synced:
            refuse_unsynced(path, checked.shards)
        _same_thresholds(schema, checked.shards)
        rows = sum(shard.rows for shard, _ in checked.shards)
        return rows, stream_of(info, checked.shards)
    finally:
        checked.shards.clear()  # a sink keeps no mappings, also on refusal


def _same_thresholds(schema: ScoreSchema, shards: list) -> None:
    if not shards:
        return
    stored = shards[-1][0].checksums
    for name, wanted in schema.threshold_checksums().items():
        if stored.get(name) != wanted:
            raise ValueError(
                f"the thresholds of head {name.removeprefix('thresholds.')!r} differ from "
                "those in the store; a run with other thresholds would break the chain of "
                "its shards. Write a new store, or change the thresholds when you read it "
                "(ScoreStore.predict_batches(thresholds=...))"
            )
