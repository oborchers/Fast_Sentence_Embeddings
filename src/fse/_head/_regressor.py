#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""``fse.Regressor``: an encoder and a linear regression head (``predict``)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from numpy import ndarray

from fse._validation import PathLike, PositiveInt, validate_arguments

from . import _run, _sklearn
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
        ``Normalizer``, an optional ``StandardScaler`` and the regressor) on the rows of
        ``encoder``; reads its attributes and imports nothing of scikit-learn."""
        found = _sklearn.read(model, "regression")
        return cls._build(
            encoder,
            found.weight,
            found.bias,
            normalized=_sklearn.flag(found, normalized),
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

    @validate_arguments
    def predict(
        self,
        texts: Any,
        *,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        sink: Any = None,
    ) -> ndarray | None:
        """The values of every text (or row of a ``fse.VectorStore``), float32 (n,) for one
        target, else (n, targets); with a ``sink`` the rows go there."""
        return _run.infer(self, texts, "values", workers, backend, sink)

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
    ) -> Iterator[PredictedBatch]:
        """Predict a stream batch by batch, as ``Classifier.predict_batches``."""
        return _run.batches(self, batches, output, return_vectors, workers, backend, sink)
