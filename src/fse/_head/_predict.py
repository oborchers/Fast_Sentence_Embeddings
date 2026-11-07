#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The methods of ``Classifier`` that read texts: ``predict``, ``predict_proba``,
``decision_function`` and ``predict_batches``, with what the length rule cut on request.

They live in a mixin so that ``_classifier.py`` stays within its limit; it uses only what a
classifier offers (``_check_proba``) and does not import the classifier.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, overload

from numpy import ndarray

from fse._runtime import LengthReport, LengthSettings, Pool, workers_of
from fse._validation import PositiveInt, validate_arguments

from . import _lone_run, _resume_run, _run
from ._outputs import PredictedBatch

Level = Literal["summary", "per_row"]


class _Predicting:
    """``predict`` and its relatives, for ``Classifier``."""

    __slots__ = ()

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
    ) -> ndarray: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        lengths: Level,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: None = None,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: Any = None,
    ) -> tuple[ndarray, LengthReport]: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: Any = None,
        score_sink: Any,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
    ) -> tuple[None, LengthReport] | None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: Any,
        score_sink: None = None,
        doc_ids: Any = None,
        shard: tuple[int, int] | None = None,
    ) -> tuple[None, LengthReport] | None: ...

    @validate_arguments
    def predict(
        self,
        texts: Any,
        *,
        workers: PositiveInt | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = None,
        shard: Any = None,
    ) -> Any:
        """The label of each text or VectorStore row. For multilabel heads, a boolean
        (n, k) indicator in classes_ order, true strictly above each stored cutoff.
        Other classifiers return one class per row as before.

        ``lengths="summary"`` or ``"per_row"`` returns ``(labels, fse.LengthReport)``: what
        the length rule of the encoder cut (texts only; see ``Encoder.encode`` for
        ``lengths``, ``count_truncated`` and ``on_truncation``).

        With a ``score_sink`` (``fse.ScoreSink(path, classifier, name=..., resume=True)``) or
        a ``vector_sink`` made with ``resume=True`` the call resumes and returns ``None``;
        see ``Heads.predict`` for ``vector_sink``, ``score_sink``, ``doc_ids``, ``shard`` and
        ``return_vectors``."""
        workers = workers_of(workers, pool)  # before anything is read
        method = "Classifier.predict"
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, method)
        if _resume_run.wanted(
            vector_sink, score_sink, doc_ids, shard, return_vectors=return_vectors
        ):
            return _resume_run.predict_lone(
                self, texts, return_vectors, workers, backend,
                vector_sink, score_sink, doc_ids, shard, settings,
            )  # fmt: skip
        return _run.infer(self, texts, "labels", workers, backend, None, settings, method)

    @overload
    def predict_proba(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> ndarray | None: ...

    @overload
    def predict_proba(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: Level,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> tuple[ndarray | None, LengthReport]: ...

    @validate_arguments
    def predict_proba(
        self,
        texts: Any,
        *,
        workers: PositiveInt | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> Any:
        """The probability of every label for every text, float32 (n, classes); with a
        ``sink`` the rows go there and nothing is returned. With ``lengths`` the pair
        ``(result, fse.LengthReport)``, the result ``None`` for a sink."""
        workers = workers_of(workers, pool)  # before anything is read
        method = "Classifier.predict_proba"
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, method)
        self._check_proba()  # type: ignore[attr-defined]
        return _run.infer(self, texts, "proba", workers, backend, sink, settings, method)

    @overload
    def decision_function(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> ndarray | None: ...

    @overload
    def decision_function(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: Level,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> tuple[ndarray | None, LengthReport]: ...

    @validate_arguments
    def decision_function(
        self,
        texts: Any,
        *,
        workers: PositiveInt | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
    ) -> Any:
        """The logits of every text as float32. Multilabel always returns (n, k),
        including k = 1. Single-label heads return (n,) for one output, else (n, k). With
        ``lengths`` the pair ``(result, fse.LengthReport)``, the result ``None`` for a sink."""
        workers = workers_of(workers, pool)  # before anything is read
        method = "Classifier.decision_function"
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, method)
        return _run.infer(self, texts, "decision", workers, backend, sink, settings, method)

    @overload
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Literal["labels", "proba", "decision"] = "labels",
        return_vectors: bool = False,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = False,
        lengths: None = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
    ) -> Iterator[PredictedBatch]: ...

    @overload
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Literal["labels", "proba", "decision"] = "labels",
        return_vectors: bool = False,
        workers: int | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = False,
        lengths: Level,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
    ) -> Iterator[tuple[PredictedBatch, LengthReport]]: ...

    @validate_arguments
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Literal["labels", "proba", "decision"] = "labels",
        return_vectors: bool = False,
        workers: PositiveInt | None = None,
        pool: Pool | None = None,
        backend: str = "cpu",
        sink: Any = None,
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: Any = False,
        lengths: Any = None,
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
    ) -> Iterator[Any]:
        """Predict a stream batch by batch, as ``Encoder.encode_batches``: ``batches``
        yields finite lists of texts, or is a ``fse.VectorStore`` (one batch per block of
        rows). Every batch gives one ``PredictedBatch`` with its ``outputs``; with
        ``return_vectors=True`` also its vectors, the encoder's own rows (one pooling gives
        both). What the ``sink`` takes depends on ``return_vectors``::

            batches      return_vectors  the sink takes                 outputs       vectors
            texts        False           the outputs (not labels)       None (sink)   None
            texts        True            the vectors of the encoder     the outputs   None (sink)
            VectorStore  False           the outputs (not labels)       None (sink)   None
            VectorStore  True            raises ValueError

        ``sink`` keeps this role and stores no document IDs; a shard sink of fse with
        ``doc_ids=True`` is refused, a sink of your own with ``has_doc_ids`` True takes the
        rows without IDs, as in 2.0. For vectors, scores and document IDs use the keywords
        of ``fse.Heads.predict_batches``, which work the same here and exclude ``sink``:

        - ``vector_sink`` takes the vectors of the encoder (needs ``return_vectors=True``;
          the head on raw means then takes the unfolded path, see ``path``);
        - ``score_sink`` is a ``fse.ScoreSink(path, model, name=...)`` of this head; the
          outputs of the batch stay in it, ``durable_rows`` is the minimum of both sinks;
        - ``doc_ids=True`` gives the batches as ``(texts, doc_ids)`` pairs, stored with the
          rows of every sink that has ``doc_ids=True``.

        These three take texts; a ``VectorStore`` input with one of them raises. With
        ``lengths="summary"`` or ``"per_row"`` every item is ``(PredictedBatch,
        fse.LengthReport)``, the report of the rows of that batch (texts only). Checked
        before any batch is read.

        ``positions=True`` resumes an endless stream that can be read again from a position,
        into sinks made with ``resume=True`` (see ``fse.Heads.predict_batches``); it excludes
        ``sink``."""
        workers = workers_of(workers, pool)  # before anything is read
        method = "Classifier.predict_batches"
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, method)
        if output == "proba":
            self._check_proba()  # type: ignore[attr-defined]
        return _lone_run.batches(
            self,
            batches,
            output,
            return_vectors,
            workers,
            backend,
            sink,
            vector_sink,
            score_sink,
            doc_ids,
            settings,
            positions,
        )
