"""Prediction methods split out to keep the concrete model within its file limit."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal, overload

from numpy import ndarray

from fse._runtime import (
    LengthReport,
    LengthSettings,
    Pool,
    workers_of,
)
from fse._validation import F, PositiveInt, validate_arguments

from . import _heads_run, _resume_run
from ._outputs import HeadsBatch


def _validated(function: F) -> F:
    function.__qualname__ = "Heads." + function.__name__
    return validate_arguments(function)


class _HeadsPredicting:
    """Prediction methods inherited by the concrete model."""

    __slots__ = ()

    @overload
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: int | None = None,
        backend: str = "cpu",
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> Mapping[str, ndarray]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: int | None = None,
        backend: str = "cpu",
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: Literal[True],
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[Mapping[str, ndarray], ndarray]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: int | None = None,
        backend: str = "cpu",
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> Mapping[str, ndarray] | tuple[Mapping[str, ndarray], ndarray]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: int | None = None,
        backend: str = "cpu",
        lengths: Literal["summary", "per_row"],
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[Mapping[str, ndarray], LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: int | None = None,
        backend: str = "cpu",
        lengths: Literal["summary", "per_row"],
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: Literal[True],
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[Mapping[str, ndarray], ndarray, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: int | None = None,
        backend: str = "cpu",
        lengths: Literal["summary", "per_row"],
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> (
        tuple[Mapping[str, ndarray], LengthReport]
        | tuple[Mapping[str, ndarray], ndarray, LengthReport]
    ): ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: int | None = None,
        backend: str = "cpu",
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: Any = None,
        score_sink: Any,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> tuple[None, LengthReport] | None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: int | None = None,
        backend: str = "cpu",
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: Any,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> tuple[None, LengthReport] | None: ...

    @_validated
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> Any:
        """The outputs of every head for every text, by name, in head order. By default each
        head gives what its own ``predict`` gives (labels, or the values of a regressor).
        ``output`` is an output name for all heads (``labels``, ``proba``, ``decision``,
        ``values``) or a mapping of head names to names; unnamed heads keep their default.
        All checks run before a text is read.

        ``lengths="summary"`` or ``"per_row"`` returns ``(outputs, fse.LengthReport)``: what
        the length rule of the encoder cut. See ``Encoder.encode`` for ``lengths``,
        ``count_truncated`` and ``on_truncation``.

        With a ``score_sink`` or a ``vector_sink`` made with ``resume=True`` (``fse.ScoreSink``,
        ``fse.ShardSink``) the call resumes and returns ``None`` (with ``lengths`` the pair
        ``(None, fse.LengthReport)``): ``texts`` must be read again (a list, tuple or array of
        str, token lists or a path of a text file), each sink compares the identity of the
        source with its store and refuses another source, partition or head, and only the
        rows the stores do not hold yet are run, bit for bit as an uninterrupted run would.
        The stores of a call may hold different numbers of rows: the longer one skips what it
        holds, the shorter one is caught up first. ``shard=(i, n)`` takes part ``i`` of ``n`` of
        the rows (whole blocks of 16,384). ``doc_ids`` are the document IDs of the texts, for
        sinks made with ``doc_ids=True``. ``return_vectors=True`` takes the path of the heads
        that gives their vectors too (see ``paths``), as a store written with a
        ``vector_sink`` needs; without a ``vector_sink`` the vectors are dropped. A report of
        ``lengths`` covers the rows run by this call, its ``first_row`` is their row in the
        source. The outputs are not returned, so the memory of the call does not grow with the
        rows of the source; ``output`` is checked and changes nothing the stores hold.
        Without sinks, ``return_vectors=True`` returns ``(outputs, vectors)`` or, with
        ``lengths``, ``(outputs, vectors, fse.LengthReport)``. Otherwise the result is
        ``outputs`` or ``(outputs, fse.LengthReport)``. Vectors are the encoder's float32
        rows, bit for bit as in ``predict_batches(return_vectors=True)``. Raw-mean heads
        then run unfolded. Standalone models and VectorStore inputs cannot return vectors.
        Vectors need ``4*n*dim`` bytes, about ``8*n*dim`` at peak for unknown row counts.
        Use ``predict_batches`` for large or unbounded sources. Resume sinks return None
        (with lengths, ``(None, fse.LengthReport)``); a score sink alone drops the vectors."""
        threads = workers_of(workers, pool)  # before anything is read
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, "Heads.predict")
        if _resume_run.wanted(vector_sink, score_sink, doc_ids, shard):
            return _resume_run.predict_heads(
                self, texts, output, return_vectors, threads, backend,
                vector_sink, score_sink, doc_ids, shard, settings,
            )  # fmt: skip
        return _heads_run.infer(
            self, texts, output, threads, backend, settings, return_vectors=return_vectors
        )

    @_validated
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Any = None,
        return_vectors: bool = False,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = False,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Iterator[HeadsBatch]:
        """``predict`` for a stream, batch by batch: ``batches`` yields finite lists of texts
        or Arrow arrays. Every batch gives one ``HeadsBatch`` with the ``outputs`` of every
        head and, with ``return_vectors=True``, the ``vectors`` (the rows of the encoder; the
        heads on raw means then take the unfolded path, see ``paths``). Checked before any
        batch is read.

        With a ``score_sink`` (``fse.ScoreSink``) the scores of every head are also written
        to a score store, whatever ``output`` returns; the call attaches the run to the sink
        before the first batch is read, so a head whose path differs from that of the store
        is refused. Each batch then reports ``durable_rows``, the rows of the run that are
        durable; a consumer of a queue confirms its input up to there, and up to
        ``sink.durable_rows`` once the iteration ended (the sink is flushed then).

        A ``vector_sink`` takes the vectors of the encoder, the rows of ``return_vectors=True``
        (which it needs: the heads on raw means then take the unfolded path, see ``paths``).
        It is a ``fse.ShardSink`` or any ``fse.Sink``, alone or with a ``score_sink`` in a
        second folder: one pass writes both, with the same ``doc_ids``. ``durable_rows`` is
        then the minimum of the counts of both sinks. The two folders are not one store: no
        file ties them, each is a valid prefix on its own after a crash, and rows beyond the
        confirmed ones may be in one folder only; a replay writes them again (at least once).
        Give both sinks the same ``shard_rows``; document IDs are not unique keys. Two shard
        stores that hold unequal numbers of rows take a run only with ``doc_ids``. With
        ``doc_ids=True`` the batches are ``(texts, doc_ids)`` pairs as for
        ``Encoder.encode_batches(doc_ids=True)``, stored with the rows; the sink must have
        been made with ``doc_ids=True``. Stopping early does not flush: close the sink.

        With ``lengths="summary"`` or ``"per_row"`` each batch has a ``HeadsBatch.lengths``,
        the ``fse.LengthReport`` of its rows (see ``Encoder.encode``).

        ``positions=True`` resumes an endless stream that can be read again from a position
        (a Kafka offset, a byte offset). The sinks are made with ``resume=True`` and every
        batch is ``(texts, position)``, or ``(texts, doc_ids, position)`` with
        ``doc_ids=True``; ``position`` is a str of 1 to 65,536 UTF-8 bytes that names the first
        record of the batch (fse stores it and never reads it). Each shard keeps the position
        of the batch it ends in. After a crash ``sink.source_position`` is ``(P, skip)``: seek
        the source to ``P`` and give its batches again; the first ``skip`` records, which the
        store holds already, are dropped. With two sinks that hold different numbers of rows,
        seek by the sink that holds fewer. Every batch of the stream gives one result: its
        ``first_row`` is the replay row (counted from the seek) of the first record computed,
        ``n_rows`` the records computed (0 for a batch the stores hold), the outputs and
        vectors are those of the records computed, and ``durable_rows`` the replay rows that
        are durable in every sink, so a consumer commits its input up to there.

        A single head takes the same keywords in its own ``predict_batches``, without a
        group."""
        threads = workers_of(workers, pool)  # before anything is read
        settings = LengthSettings.of(
            lengths, count_truncated, on_truncation, "Heads.predict_batches"
        )
        return _heads_run.batches(
            self,
            batches,
            output,
            return_vectors,
            threads,
            backend,
            vector_sink,
            score_sink,
            doc_ids,
            settings,
            positions,
        )
