#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``encode`` and ``encode_batches`` of the encoder, with what the length rule cut on request.

They live in a mixin so that ``_encoder.py`` stays within its limit; it uses only what the
encoder offers (``_pipeline``, ``_spec``, ``fingerprint``, ``output_dim`` and ``tokenizer``)
and does not import the encoder.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Literal, overload

from numpy.typing import NDArray

from fse import _codes
from fse._runtime import (
    LengthCollector,
    LengthReport,
    LengthSettings,
    Pool,
    QuantizedRows,
    Sink,
    check_workers,
    rule_of,
    workers_of,
)
from fse._runtime._lengths import LengthLevel

from . import _batches, _positions, _resume
from ._batches import EncodedBatch, check_sink, flush, run_into
from ._doc_ids import check_id_mode
from ._sources import opened


class _Encoding:
    """The methods of ``Encoder`` that turn texts into vectors."""

    __slots__ = ()

    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: Literal["float32", "float16"] | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        doc_ids: Any = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        shard: None = None,
        pool: Pool | None = None,
    ) -> NDArray[Any]: ...
    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: Literal["int8"],
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        doc_ids: Any = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        shard: None = None,
        pool: Pool | None = None,
    ) -> QuantizedRows: ...
    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Sink,
        doc_ids: Any = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> None: ...
    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: Literal["float32", "float16"] | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        doc_ids: Any = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        shard: None = None,
        pool: Pool | None = None,
    ) -> tuple[NDArray[Any], LengthReport]: ...
    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: Literal["int8"],
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        doc_ids: Any = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        shard: None = None,
        pool: Pool | None = None,
    ) -> tuple[QuantizedRows, LengthReport]: ...
    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Sink,
        doc_ids: Any = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> tuple[None, LengthReport]: ...
    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Sink | None = None,
        doc_ids: Any = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> NDArray[Any] | QuantizedRows | None: ...
    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Sink | None = None,
        doc_ids: Any = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> tuple[NDArray[Any] | QuantizedRows | None, LengthReport]: ...

    def encode(
        self,
        texts: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        doc_ids: Any = None,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> Any:
        """The vectors of ``texts``, one row per sentence, as ``dtype`` (float32 by
        default): float32 or float16 rows, or int8 ``QuantizedRows``. Without a sink the
        whole result is held in memory. With a ``sink`` (``fse.Sink``, such as
        ``fse.ShardSink``), the rows go there and nothing is returned; ``dtype`` is then
        the sink's (another one raises) and its ``flush`` runs after the last row.

        ``texts`` is a list, tuple or numpy array of str, or of token lists for the
        external tokenizer; a ``pathlib.Path`` (or other
        ``os.PathLike``) of a text file, one sentence per line (a str is one sentence,
        never a path: pass ``[text]`` for one sentence); or any other iterable of str, a
        stream read once.

        ``workers`` is the number of threads (None: the usable CPUs, or the workers of
        ``pool``, which it may not exceed). ``pool`` is an ``fse.Pool`` to share between
        calls; none starts threads for this call. ``backend`` names the backend that runs
        the kernels (``"cpu"``). ``doc_ids`` is one integer ID per text (a list or a
        one-dimensional int64 array), only with a sink made with ``doc_ids=True``, which
        needs them; keep the array unchanged until the call returns.

        ``lengths="summary"`` or ``"per_row"`` returns ``(result, fse.LengthReport)``: what
        the length rule of the encoder (``max_tokens`` and ``max_chars``) cut, as sums or
        also per row (with a sink the result is ``None``). ``count_truncated=True`` adds how
        many tokens and characters were cut. ``on_truncation="raise"`` raises
        ``fse.TruncationError`` for the first sentence the rule would cut instead of cutting
        it. A flag costs at most one more word of a sentence that reaches
        ``max_tokens``; counting tokenizes the rest of every cut sentence. Without these keywords nothing is recorded and nothing changes.

        With a sink made with ``resume=True`` (``fse.ShardSink``) the call resumes: ``texts``
        must be read again (a list, tuple or array of str, token lists or a path), the sink
        compares its fingerprint with the store's and refuses another source or encoder, and
        only the rows the store does not hold yet are encoded, bit for bit as an
        uninterrupted run would. ``shard=(i, n)`` takes part ``i`` of ``n`` of the rows (whole
        blocks of 16,384; the parts are disjoint and complete) so ``n`` processes can share a
        source, each writing a store of its own. A report of ``lengths`` covers the rows run
        by this call, and its ``first_row`` is their row in the source.
        """
        threads = workers_of(workers, pool)  # before anything is read
        me: Any = self  # the Encoder
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, "Encoder.encode")
        check_id_mode(sink, doc_ids)
        if _resume.wanted(sink, shard):
            return _encode_resumed(
                me, texts, dtype, threads, backend, sink, doc_ids, settings, shard
            )
        collector = _collector(settings, 0, me)
        if sink is not None:
            check_sink(sink, dtype, me.fingerprint, me.output_dim)
            pipeline = me._pipeline(backend)
            with opened(texts, me.tokenizer) as source:
                run_into(pipeline, source, sink, threads, 0, doc_ids, settings, collector)
            flush(sink)
            result = None
        else:
            dtype = _codes.check_dtype("float32" if dtype is None else dtype)
            pipeline = me._pipeline(backend)
            with opened(texts, me.tokenizer) as source:
                result = pipeline.infer(
                    source, workers=threads, dtype=dtype, lengths=settings, collector=collector
                )
        return result if collector is None else (result, collector.report())

    @overload
    def encode_batches(
        self,
        batches: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        doc_ids: bool = False,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Generator[EncodedBatch, None, None]: ...

    @overload
    def encode_batches(
        self,
        batches: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        doc_ids: bool = False,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Generator[tuple[EncodedBatch, LengthReport], None, None]: ...

    def encode_batches(
        self,
        batches: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        doc_ids: bool = False,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Generator[Any, None, None]:
        """Encode a stream batch by batch: ``batches`` yields finite lists of sentences
        (or token lists), as a queue consumer polls them, and every batch gives one
        ``EncodedBatch`` once its rows are finished. With a ``sink`` the rows go there,
        and ``durable_rows`` says how many rows of the run are durable, so a consumer
        confirms its input only up to there; without one the batch holds its rows.

        The arguments and the encoder are checked when this is called, before any batch
        is read; closing the iterator early reads no further batch. After a successful
        full iteration the sink's ``flush`` (where it has one) has run, and its own
        ``durable_rows`` then says what is durable; for a ``fse.ShardSink`` that is every
        row of the run, while the last batch may report fewer. Stopping
        early (``break``, ``close()``) does not flush: the owner of the sink closes it
        (``ShardSink.close`` publishes the rows committed so far). Without a sink every
        batch holds its rows in memory. With ``doc_ids=True`` every batch is a
        ``(texts, doc_ids)`` tuple; the sink must have document IDs, and the sink and the
        argument must agree.

        ``lengths`` makes every item the pair ``(EncodedBatch, fse.LengthReport)``, the
        report of the rows of that batch (its ``first_row`` is the run's); see
        ``encode`` for ``lengths``, ``count_truncated`` and ``on_truncation``.

        ``positions=True`` resumes an endless stream that can be read again from a position
        (a Kafka offset, a byte offset): the sink is made with ``resume=True`` and every
        batch is ``(texts, position)``, or ``(texts, doc_ids, position)`` with
        ``doc_ids=True``, where ``position`` is a str of 1 to 65,536 UTF-8 bytes that names the
        first record of the batch (fse stores it and never reads it). Each shard keeps the
        position of the batch it ends in. After a crash ``sink.source_position`` is
        ``(P, skip)``: seek the source to ``P`` and give its batches again; the first ``skip``
        records, which the store holds already, are dropped without being encoded. Every item
        covers one batch: ``first_row`` is the replay row (counted from the seek) of the first
        record computed, ``n_rows`` the records computed (0 for a batch the store holds) and
        ``durable_rows`` the replay rows that are durable, so a consumer commits its input up
        to there. A stream that ends before it delivered the records the store holds raises a
        ``ValueError``.
        """
        threads = workers_of(workers, pool)  # before anything is read
        me: Any = self  # the Encoder
        settings = LengthSettings.of(
            lengths, count_truncated, on_truncation, "Encoder.encode_batches"
        )
        _batches.check_positions(positions)
        dtype = _batches.prepare(
            sink, dtype, doc_ids, me.fingerprint, me.output_dim, threads, positions=positions
        )
        pipeline = me._pipeline(backend)
        if positions:
            _positions.start(sink)
            return _positions.batches(
                pipeline,
                iter(batches),
                me.tokenizer,
                sink=sink,
                workers=threads,
                doc_ids=doc_ids,
                lengths=settings,
                rule=rule_of(me._spec.tokenizer),
            )
        return _batches.batches(
            pipeline,
            iter(batches),
            me.tokenizer,
            dtype=dtype,
            workers=threads,
            sink=sink,
            doc_ids=doc_ids,
            lengths=settings,
            rule=rule_of(me._spec.tokenizer),
        )


def _encode_resumed(
    me: Any,
    texts: Any,
    dtype: str | None,
    workers: Any,
    backend: str,
    sink: Any,
    doc_ids: Any,
    settings: LengthSettings | None,
    shard: Any,
) -> Any:
    """``encode`` into a sink made with ``resume=True``: the rows after those it holds."""
    part = _resume.check_arguments(texts, sink, shard)
    check_workers(workers)  # also where no row is left to run
    check_sink(sink, dtype, me.fingerprint, me.output_dim, resumed=True)
    pipeline = me._pipeline(backend)
    with _resume.prepared(texts, sink, part, doc_ids, me.tokenizer) as run:
        collector = _collector(settings, run.first_row, me)
        if run.rows:
            try:
                run_into(pipeline, run.source, sink, workers, 0, run.ids, settings, collector)
            except Exception as error:
                _resume.add_row_note(error, run.first_row)
                raise
            run.check_unchanged()
    flush(sink)
    return None if collector is None else (None, collector.report())


def _collector(
    settings: LengthSettings | None, first_row: int, encoder: Any
) -> LengthCollector | None:
    """The collector of a call that asked for a report (a level of ``lengths``)."""
    if settings is None:
        return None
    return settings.collector(first_row, rule_of(encoder._spec.tokenizer))
