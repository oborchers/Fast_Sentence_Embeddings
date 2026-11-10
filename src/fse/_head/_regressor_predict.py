"""Prediction methods split out to keep the concrete model within its file limit."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from fse._runtime import LengthReport, LengthSettings, Pool, Sink, workers_of
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
        sink: None = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> NDArray[np.float32]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Sink,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[NDArray[np.float32], LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Sink,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[None, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> NDArray[np.float32] | None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: Literal[True],
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: bool = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.float32]] | None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: Literal[False] = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[NDArray[np.float32] | None, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: Literal[True],
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: bool = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
        pool: Pool | None = None,
    ) -> (
        tuple[NDArray[np.float32] | None, LengthReport]
        | tuple[NDArray[np.float32], NDArray[np.float32], LengthReport]
    ): ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: bool = False,
        vector_sink: Sink | None = None,
        score_sink: Sink,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: bool = False,
        vector_sink: Sink | None = None,
        score_sink: Sink,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> tuple[None, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: bool = False,
        vector_sink: Sink,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: bool = False,
        vector_sink: Sink,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> tuple[None, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: None = None,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: bool = False,
        vector_sink: Sink | None = None,
        score_sink: Sink | None = None,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.float32]] | None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: None = None,
        lengths: LengthLevel,
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        return_vectors: bool = False,
        vector_sink: Sink | None = None,
        score_sink: Sink | None = None,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
        pool: Pool | None = None,
    ) -> (
        tuple[NDArray[np.float32], LengthReport]
        | tuple[NDArray[np.float32], NDArray[np.float32], LengthReport]
        | tuple[None, LengthReport]
    ): ...

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
        """The values of every text (or row of a ``fse.VectorStore``), float32 (n,) for a flat
        head (one target from a one-dimensional ``coef_``), else (n, targets); with a ``sink`` the rows go there. ``lengths="summary"``
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
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Generator[PredictedBatch, None, None]: ...

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
        count_truncated: bool = False,
        on_truncation: Literal["keep", "raise"] = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Generator[tuple[PredictedBatch, LengthReport], None, None]: ...

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
    ) -> Generator[Any, None, None]:
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
