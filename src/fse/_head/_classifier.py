#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""``fse.Classifier``: an encoder and a linear classification head, as scikit-learn names
its outputs (``predict``, ``predict_proba``, ``decision_function``, ``classes_``)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from numpy import ndarray

from fse._validation import PathLike, PositiveInt, validate_arguments

from . import _run, _sklearn
from ._content import classes
from ._model import LinearModel
from ._outputs import PredictedBatch


class Classifier(LinearModel):
    """Predicts labels of texts with an encoder and a linear head ``z = A x + b`` on its
    rows: ``labels[1]`` if ``z > 0`` for one output, else the label of the largest logit.

    Make one with ``Classifier.from_arrays`` or ``Classifier.load``; fse trains nothing. A head trained
    on raw means (an encoder with ``normalize=False``) folds into the table and predicts
    faster than ``encode``; one trained on normalized vectors runs on the full mean.
    """

    __slots__ = ()
    TASK = "classification"

    @classmethod
    @validate_arguments
    def from_sklearn(cls, encoder: Any, model: Any, *, normalized: bool | None = None):
        """The head of a fitted scikit-learn classifier (or a Pipeline of an optional
        ``Normalizer``, an optional ``StandardScaler`` and the classifier) on the rows of
        ``encoder``, with the probabilities of the installed version's rule. Reads its
        attributes and imports nothing of scikit-learn. ``normalized``: whether it was
        trained on normalized vectors; by default the encoder's setting (a ``Normalizer``
        implies True)."""
        found = _sklearn.read(model, "classification")
        return cls._build(
            encoder,
            found.weight,
            found.bias,
            normalized=_sklearn.flag(found, normalized),
            link=found.link,
            labels=found.labels,
            source=found.source,
        )

    @classmethod
    @validate_arguments
    def from_arrays(
        cls,
        encoder: Any,
        weight: Any,
        bias: Any = None,
        *,
        labels: Any,
        link: Literal["auto", "sigmoid", "softmax", "ovr", "none"] = "auto",
        normalized: bool | None = None,
    ):
        """The head ``weight`` (k, d) or (d,) and ``bias`` (k, a scalar or None) on the rows
        of ``encoder``, d its ``output_dim``: ``labels`` are k labels, or two for k = 1
        (negative first, as ``classes_``), all int or all str. ``link``: ``"sigmoid"`` (k =
        1), ``"softmax"`` or ``"ovr"`` (k >= 2), ``"none"`` (no probabilities); ``"auto"``
        is sigmoid for one output and softmax else. ``normalized``: whether the head was
        trained on normalized rows, by default the encoder's setting."""
        return cls._build(encoder, weight, bias, normalized=normalized, link=link, labels=labels)

    @classmethod
    @validate_arguments
    def load(cls, path: PathLike, mmap: bool = False):  # -> Classifier
        """Open a classifier saved with ``save``, complete or standalone."""
        return cls._load(path, mmap)

    @property
    def classes_(self) -> ndarray:
        """The labels in output order (int64 or str)."""
        assert self._content.labels is not None
        return classes(self._content.labels)

    @validate_arguments
    def predict(
        self, texts: Any, *, workers: PositiveInt | None = None, backend: str = "cpu"
    ) -> ndarray:
        """The label of every text (as ``Encoder.encode`` takes them), as an array of
        ``classes_``."""
        return _run.infer(self, texts, "labels", workers, backend, None)

    @validate_arguments
    def predict_proba(
        self,
        texts: Any,
        *,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        sink: Any = None,
    ) -> ndarray | None:
        """The probability of every label for every text, float32 (n, classes); with a
        ``sink`` the rows go there and nothing is returned."""
        self._check_proba()
        return _run.infer(self, texts, "proba", workers, backend, sink)

    @validate_arguments
    def decision_function(
        self,
        texts: Any,
        *,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        sink: Any = None,
    ) -> ndarray | None:
        """The logits of every text, float32 (n,) for one output, else (n, k)."""
        return _run.infer(self, texts, "decision", workers, backend, sink)

    @validate_arguments
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Literal["labels", "proba", "decision"] = "labels",
        return_vectors: bool = False,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        sink: Any = None,
    ) -> Iterator[PredictedBatch]:
        """Predict a stream batch by batch, as ``Encoder.encode_batches``: ``batches``
        yields finite lists of texts. Every batch gives one ``PredictedBatch`` with its ``outputs``; with
        ``return_vectors=True`` also its vectors, the encoder's own rows (one pooling gives
        both), and a ``sink`` takes the vectors; without it a ``sink`` takes the outputs
        (not labels). Checked before any batch is read."""
        if output == "proba":
            self._check_proba()
        return _run.batches(self, batches, output, return_vectors, workers, backend, sink)

    def _check_proba(self) -> None:
        if self._content.link is None:
            source = f" ({self._source})" if self._source else ""
            raise ValueError(
                f"this classifier{source} has no probabilities (link none); use decision_function"
            )
