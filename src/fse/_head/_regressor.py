#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.Regressor``: an encoder and a linear regression head (``predict``)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, overload

from numpy import ndarray

from fse._encoder import _pretrained
from fse._runtime import LengthReport, LengthSettings, Pool, workers_of
from fse._validation import PathLike, PositiveInt, validate_arguments

from . import _lone_run, _model2vec_files, _resume_run, _run, _sklearn
from ._model import LinearModel
from ._outputs import PredictedBatch


class Regressor(LinearModel):
    """Predicts values of texts with an encoder and a linear head ``z = A x + b`` on its
    rows, one value per target. Make one with ``Regressor.from_sklearn``,
    ``Regressor.from_arrays`` or ``Regressor.load``; fse trains nothing. A head trained on
    raw means folds into the table.
    """

    __slots__ = ()
    TASK = "regression"

    @classmethod
    @validate_arguments
    def from_sklearn(cls, encoder: Any, model: Any, *, normalized: bool | None = None):
        """The head of a fitted scikit-learn linear regressor (or a Pipeline of an optional
        ``StandardScaler`` and the regressor) on the rows of
        ``encoder``; reads its attributes and imports nothing of scikit-learn."""
        found = _sklearn.read(model, "regression")
        return cls._build(
            encoder,
            found.weight,
            found.bias,
            normalized=normalized,
            source=found.source,
            flat=found.flat,
        )

    @classmethod
    @validate_arguments
    def from_arrays(
        cls, encoder: Any, weight: Any, bias: Any = None, *, normalized: bool | None = None
    ):
        """The head ``weight`` (targets, d) or (d,) and ``bias`` (targets, a scalar or None)
        on the rows of ``encoder``; ``normalized`` as for ``Classifier.from_arrays``."""
        return cls._build(encoder, weight, bias, normalized=normalized)

    @classmethod
    @validate_arguments
    def load(cls, path: PathLike, mmap: bool = False):  # -> Regressor
        """Open a regressor saved with ``save``, complete or standalone."""
        return cls._load(path, mmap)

    @classmethod
    @validate_arguments
    def from_pretrained(
        cls,
        repo_or_path: PathLike,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: PathLike | None = None,
        local_files_only: bool = False,
        mmap: bool = False,
    ):  # -> Regressor
        """The regressor of a local ``.safetensors`` file or of an fse folder written by
        ``save_pretrained``, local or a Hugging Face Hub repository (``namespace/name``) at
        ``revision`` (default branch if None), checked against its commit; ``mmap`` as for
        ``load``, for local files."""
        files = _pretrained.files(
            repo_or_path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        _model2vec_files.refuse_for_regressor(files)
        return cls._from_files(files, mmap)

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
    ) -> ndarray | None: ...

    @overload
    def predict(
        self,
        texts: Any,
        *,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
        lengths: Literal["summary", "per_row"],
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        return_vectors: bool = False,
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

    @validate_arguments
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
        ``score_sink``, ``doc_ids``, ``shard`` and ``return_vectors``."""
        threads = workers_of(workers, pool)  # before anything is read
        method = "Regressor.predict"
        settings = LengthSettings.of(lengths, count_truncated, on_truncation, method)
        if _resume_run.wanted(
            vector_sink, score_sink, doc_ids, shard, return_vectors=return_vectors
        ):
            if sink is not None:
                raise ValueError(
                    "sink= keeps its role: the values. Give the vectors as vector_sink= and the "
                    "scores as score_sink=, without sink="
                )
            return _resume_run.predict_lone(
                self, texts, return_vectors, threads, backend,
                vector_sink, score_sink, doc_ids, shard, settings,
            )  # fmt: skip
        return _run.infer(self, texts, "values", threads, backend, sink, settings, method)

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
        lengths: Literal["summary", "per_row"],
        count_truncated: Any = False,
        on_truncation: Any = "keep",
        positions: bool = False,
        pool: Pool | None = None,
    ) -> Iterator[tuple[PredictedBatch, LengthReport]]: ...

    @validate_arguments
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
