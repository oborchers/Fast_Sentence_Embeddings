"""Prediction methods split out to keep the concrete model within its file limit."""

from __future__ import annotations

from typing import Any, overload

from numpy import ndarray

from fse._runtime import LengthReport, LengthSettings, Pool, workers_of
from fse._runtime._lengths import LengthLevel
from fse._validation import F, PositiveInt, validate_arguments

from . import _run


def _validated(function: F) -> F:
    function.__qualname__ = "_Predicting." + function.__name__
    return validate_arguments(function)


class _Scoring:
    """Prediction methods inherited by the concrete model."""

    __slots__ = ()

    @overload
    def predict_proba(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        pool: Pool | None = None,
    ) -> ndarray | None: ...

    @overload
    def predict_proba(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: LengthLevel,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        pool: Pool | None = None,
    ) -> tuple[ndarray | None, LengthReport]: ...

    @_validated
    def predict_proba(
        self,
        texts: Any,
        *,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        pool: Pool | None = None,
    ) -> Any:
        """The probability of every label for every text, float32 (n, classes); with a
        ``sink`` the rows go there and nothing is returned. With ``lengths`` the pair
        ``(result, fse.LengthReport)``, the result ``None`` for a sink."""
        threads = workers_of(workers, pool)  # before anything is read
        method = "Classifier.predict_proba"
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, method)
        self._check_proba()  # type: ignore[attr-defined]
        return _run.infer(self, texts, "proba", threads, backend, sink, settings, method)

    @overload
    def decision_function(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        pool: Pool | None = None,
    ) -> ndarray | None: ...

    @overload
    def decision_function(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: LengthLevel,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        pool: Pool | None = None,
    ) -> tuple[ndarray | None, LengthReport]: ...

    @_validated
    def decision_function(
        self,
        texts: Any,
        *,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        pool: Pool | None = None,
    ) -> Any:
        """The logits of every text as float32. Multilabel always returns (n, k),
        including k = 1. Single-label heads return (n,) for one output, else (n, k). With
        ``lengths`` the pair ``(result, fse.LengthReport)``, the result ``None`` for a sink."""
        threads = workers_of(workers, pool)  # before anything is read
        method = "Classifier.decision_function"
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, method)
        return _run.infer(self, texts, "decision", threads, backend, sink, settings, method)
