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

from collections.abc import Iterator
from typing import Any, Literal, overload

from numpy import ndarray

from fse import _codes
from fse._runtime import LengthCollector, LengthReport, LengthSettings, QuantizedRows, rule_of

from . import _batches
from ._batches import EncodedBatch, check_sink, flush, run_into
from ._doc_ids import check_id_mode
from ._sources import opened

Level = Literal["summary", "per_row"]


class _Encoding:
    """The methods of ``Encoder`` that turn texts into vectors."""

    __slots__ = ()

    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        doc_ids: Any = None,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> ndarray | QuantizedRows | None: ...

    @overload
    def encode(
        self,
        texts: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        doc_ids: Any = None,
        lengths: Level,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> tuple[ndarray | QuantizedRows | None, LengthReport]: ...

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

        ``lengths="summary"`` or ``"per_row"`` returns ``(result, fse.LengthReport)``: what
        the length rule of the encoder (``max_tokens`` and ``max_chars``) cut, as sums or
        also per row (with a sink the result is ``None``). ``count_truncated=True`` adds how
        many tokens and characters were cut. ``on_truncation="raise"`` raises
        ``fse.TruncationError`` for the first sentence the rule would cut instead of cutting
        it. A flag costs a second tokenization of a sentence that ends exactly at
        ``max_tokens``, up to one more token; counting tokenizes the rest of every cut
        sentence. Without these keywords nothing is recorded and nothing changes.
        """
        me: Any = self  # the Encoder
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, "Encoder.encode")
        check_id_mode(sink, doc_ids)
        collector = _collector(settings, 0, me)
        if sink is not None:
            check_sink(sink, dtype, me.fingerprint, me.output_dim)
            pipeline = me._pipeline(backend)
            with opened(texts, me.tokenizer) as source:
                run_into(pipeline, source, sink, workers, 0, doc_ids, settings, collector)
            flush(sink)
            result = None
        else:
            dtype = _codes.check_dtype("float32" if dtype is None else dtype)
            pipeline = me._pipeline(backend)
            with opened(texts, me.tokenizer) as source:
                result = pipeline.infer(
                    source, workers=workers, dtype=dtype, lengths=settings, collector=collector
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
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> Iterator[EncodedBatch]: ...

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
        lengths: Level,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> Iterator[tuple[EncodedBatch, LengthReport]]: ...

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
    ) -> Iterator[Any]:
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
        """
        me: Any = self  # the Encoder
        settings = LengthSettings.of(
            lengths, count_truncated, on_truncation, "Encoder.encode_batches"
        )
        dtype = _batches.prepare(sink, dtype, doc_ids, me.fingerprint, me.output_dim, workers)
        pipeline = me._pipeline(backend)
        return _batches.batches(
            pipeline,
            iter(batches),
            me.tokenizer,
            dtype=dtype,
            workers=workers,
            sink=sink,
            doc_ids=doc_ids,
            lengths=settings,
            rule=rule_of(me._spec.tokenizer),
        )


def _collector(
    settings: LengthSettings | None, first_row: int, encoder: Any
) -> LengthCollector | None:
    """The collector of a call that asked for a report (a level of ``lengths``)."""
    if settings is None or settings.lengths is None:
        return None
    return LengthCollector(settings, first_row, rule_of(encoder._spec.tokenizer))
