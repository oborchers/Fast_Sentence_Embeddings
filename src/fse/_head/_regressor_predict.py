"""Prediction methods split out to keep the concrete model within its file limit."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, overload

from numpy import ndarray

from fse._runtime import LengthReport, LengthSettings, Pool, workers_of
from fse._runtime._lengths import LengthLevel
from fse._validation import F, PositiveInt, validate_arguments

from . import _lone_run, _resume_run, _run
from ._outputs import PredictedBatch


def _validated(function: F) -> F:
    function.__qualname__ = "Regressor." + function.__name__
    return validate_arguments(function)


class _RegressorPredicting:
    """Prediction methods inherited by the concrete model."""

    __slots__ = ()

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> ndarray | None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: Literal[True],
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[ndarray, ndarray]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> ndarray | tuple[ndarray, ndarray] | None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: LengthLevel,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[ndarray | None, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: LengthLevel,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: Literal[True],
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[ndarray, ndarray, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: LengthLevel,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[ndarray | None, LengthReport] | tuple[ndarray, ndarray, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
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
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
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
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        sink: Any = None,
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
        """The values of every text (or row of a ``fse.VectorStore``), float32 (n,) for one
        target, else (n, targets); with a ``sink`` the rows go there. ``lengths="summary"``
        or ``"per_row"`` returns ``(values, fse.LengthReport)``, the values ``None`` for a
        sink: what the length rule of the encoder cut (texts only; see ``Encoder.encode``).

        With a ``score_sink`` (``fse.ScoreSink(path, regressor, name=..., resume=True)``) or
        a ``vector_sink`` made with ``resume=True`` the call resumes and returns ``None``;
        ``sink=`` is then not given. See ``Heads.predict`` for ``vector_sink``,
        ``score_sink``, ``doc_ids``, ``shard`` and ``return_vectors``.

        Without sinks, ``return_vectors=True`` returns ``(values, vectors)`` or, with
        ``lengths``, ``(values, vectors, fse.LengthReport)``. Otherwise the result is
        ``values`` or ``(values, fse.LengthReport)``. Vectors are the encoder's float32
        rows, bit for bit as in ``predict_batches(return_vectors=True)``. Raw-mean heads
        then run unfolded. Standalone models and VectorStore inputs cannot return vectors.
        Vectors need ``4*n*dim`` bytes, about ``8*n*dim`` at peak for unknown row counts.
        Use ``predict_batches`` for large or unbounded sources. Resume sinks return None
        (with lengths, ``(None, fse.LengthReport)``); a score sink alone drops the vectors.

        ``sink=`` takes values and cannot be combined with ``return_vectors=True``."""
        threads = workers_of(workers, pool)  # before anything is read
        method = "Regressor.predict"
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, method)
        if _resume_run.wanted(vector_sink, score_sink, doc_ids, shard):
            if sink is not None:
                raise ValueError(
                    "sink= keeps its role: the values. Give the vectors as vector_sink= and the "
                    "scores as score_sink=, without sink="
                )
            return _resume_run.predict_lone(
                self, texts, return_vectors, threads, backend,
                vector_sink, score_sink, doc_ids, shard, settings,
            )  # fmt: skip
        return _run.infer(
            self, texts, "values", threads, backend, sink, settings, method,
            return_vectors=return_vectors,
        )  # fmt: skip

    @overload
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Literal["values"] = "values",
        return_vectors: bool = False,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = False,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Iterator[PredictedBatch]: ...

    @overload
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Literal["values"] = "values",
        return_vectors: bool = False,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = False,
        lengths: LengthLevel,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Iterator[tuple[PredictedBatch, LengthReport]]: ...

    @_validated
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Literal["values"] = "values",
        return_vectors: bool = False,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        sink: Any = None,
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = False,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Iterator[Any]:
        """Predict a stream batch by batch, as ``Classifier.predict_batches``, with the same
        ``sink``, ``vector_sink``, ``score_sink``, ``doc_ids``, ``lengths`` and ``positions``."""
        threads = workers_of(workers, pool)  # before anything is read
        settings = LengthSettings.of(
            lengths, count_truncated, on_truncation, "Regressor.predict_batches"
        )
        return _lone_run.batches(
            self,
            batches,
            output,
            return_vectors,
            threads,
            backend,
            sink,
            vector_sink,
            score_sink,
            doc_ids,
            settings,
            positions,
        )
