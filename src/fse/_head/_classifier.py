#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.Classifier``: an encoder and a linear classification head, as scikit-learn names
its outputs (``predict``, ``predict_proba``, ``decision_function``, ``classes_``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Literal, Self

from numpy.typing import NDArray

from fse._contract import Head, Link
from fse._encoder import _pretrained
from fse._validation import (
    ModelRef,
    PathLike,
    Temperature,
    ThresholdInput,
    validate_arguments,
)

from . import _calibration, _model2vec_files, _model2vec_objects, _sklearn
from ._content import classes
from ._model import LinearModel
from ._predict import _Predicting
from ._predict_scores import _Scoring
from ._thresholds import updated


class Classifier(_Predicting, _Scoring, LinearModel):
    """Predicts labels of texts with an encoder and a linear head ``z = A x + b`` on its
    rows. Single-label heads choose one class. Multilabel heads independently compare
    each probability with its stored cutoff, returning a boolean matrix.

    Make one with ``Classifier.from_sklearn`` (a fitted scikit-learn classifier),
    ``Classifier.from_arrays`` or ``Classifier.load``; fse trains nothing. A head trained
    on raw means (an encoder with ``normalize=False``) folds into the table and predicts
    faster than ``encode``; one trained on normalized vectors runs on the full mean.
    """

    __slots__ = ()
    TASK = "classification"

    @classmethod
    @validate_arguments
    def from_sklearn(
        cls,
        encoder: Any,
        model: Any,
        *,
        normalized: bool | None = None,
        labels: Any = None,
        thresholds: ThresholdInput = None,
    ) -> Self:
        """The head of a fitted scikit-learn classifier (or a Pipeline of an optional
        ``StandardScaler`` and the classifier) on the rows of
        ``encoder``, including a multilabel OneVsRestClassifier, with the probabilities of the installed version's rule. Reads its
        attributes and imports nothing of scikit-learn. ``normalized``: whether it was
        trained on normalized vectors; by default the encoder's setting. A Pipeline with a
        ``Normalizer`` raises: train on a normalizing encoder's vectors instead.

        ``labels`` optionally renames all classes in their existing output order,
        without changing weights. Indicator-matrix OneVsRest training usually stores
        numeric column indices, not application names. Multilabel import requires
        affine sigmoid binary estimators (or constant outputs). ``thresholds`` accepts
        the same forms as from_arrays, referring to the final class names.

        A ``CalibratedClassifierCV(method="sigmoid", ensemble=False)`` (binary or one-vs-rest,
        its base model a linear classifier, alone or after a ``StandardScaler``, also frozen)
        is imported with its Platt sigmoids folded into the weights and bias in float64, then
        rounded once to float32. Its head has the label rule ``linear@2``: the first class
        of largest probability, as ``predict`` of scikit-learn, so ``predict`` costs more
        than for an uncalibrated head. fse stays within derived bounds of scikit-learn's
        probabilities, except in the deep underflow of one-vs-rest (all logits below -104,
        where fse returns the uniform row and the first class, and scikit-learn goes on in
        float64). ``ensemble=True`` or ``"auto"``, ``method="isotonic"`` or ``"temperature"``,
        and calibrated estimators inside a ``OneVsRestClassifier`` raise."""
        found = _sklearn.read(model, "classification")
        return cls._build(
            encoder,
            found.weight,
            found.bias,
            normalized=normalized,
            link=found.link,
            labels=found.labels if labels is None else labels,
            source=found.source,
            thresholds=thresholds,
            rule=found.rule,
            note=found.note,
        )

    @classmethod
    @validate_arguments
    def from_arrays(
        cls,
        encoder: Any,
        weight: Any,
        bias: Any = None,
        *,
        labels: Any = None,  # required; a default keeps pydantic 2.5 from refusing the order
        link: Literal["auto", "sigmoid", "softmax", "ovr", "none", "multilabel"] = "auto",
        normalized: bool | None = None,
        thresholds: ThresholdInput = None,
        temperature: Temperature | None = None,
    ) -> Self:
        """The head ``weight`` (k, d) or (d,) and ``bias`` (k, a scalar or None) on the rows
        of ``encoder``, d its ``output_dim``: ``labels`` are k labels for multilabel;
        other heads need two for k = 1 and k otherwise
        (negative first, as ``classes_``), all int or all str. ``link``: ``"sigmoid"`` (k =
        1), ``"softmax"`` or ``"ovr"`` (k >= 2), ``"none"`` (no probabilities); ``"auto"``
        is sigmoid for one output and softmax else. ``normalized``: whether the head was
        trained on normalized rows, by default the encoder's setting.

        ``multilabel`` gives one independent sigmoid per label, including k = 1.
        ``thresholds``: None (0.5 each), a scalar, a vector, or a partial mapping from
        labels to cutoffs in [0, 1]; unnamed labels keep 0.5. Predictions are boolean
        (n, k), true strictly above the cutoff. Other links refuse thresholds.

        ``temperature`` T (finite, greater than zero) is the softmax temperature of a head
        whose logits are divided by it, ``softmax(z / T)``; only the link softmax takes it.
        fse divides ``weight`` and ``bias`` by T in float64 and rounds once to float32;
        nothing of T is stored. The label is then the first class of largest probability
        (the rule ``linear@2``, format 2), so ``predict`` costs more than without a
        temperature. Softmax of one logit ``z`` is ``sigmoid(2 z)``: for ``sigmoid(z / T)``
        give ``2 T``."""
        rule = None
        note = None
        if temperature is not None:
            weight, bias = _calibration.temperature(weight, bias, temperature, link)
            rule, note = Head.LINEAR2, _calibration.note(temperature)
        return cls._build(
            encoder,
            weight,
            bias,
            normalized=normalized,
            link=link,
            labels=labels,
            thresholds=thresholds,
            rule=rule,
            note=note,
        )

    @property
    def thresholds(self) -> Mapping[int | str, float] | None:
        """A read-only snapshot of effective F32 cutoffs in classes_ order.

        Only multilabel heads have cutoffs; every other head gives None. Values expose
        their actual stored precision.
        """
        if self._head.link is not Link.MULTILABEL:
            return None
        return MappingProxyType(self._cutoffs())

    def _cutoffs(self) -> dict[int | str, float]:
        assert self._head.thresholds is not None
        assert self._content.labels is not None
        return dict(zip(self._content.labels, map(float, self._head.thresholds), strict=True))

    @validate_arguments
    def with_thresholds(self, thresholds: ThresholdInput) -> Self:
        """A new multilabel model with changed cutoffs, also after load or standalone.

        A partial mapping keeps unnamed current values. A scalar or full vector replaces
        all values; explicit None resets all to 0.5. The original model is unchanged.
        Heads that are not multilabel have no cutoffs and raise ValueError.
        No training, tokenization or folding runs here.
        """
        if self._head.link is not Link.MULTILABEL:
            raise ValueError("with_thresholds is available only for multilabel heads")
        current = self._cutoffs()
        # Compare loaded tensors before deriving a new model, as quantize does.
        old = self._content
        assert old.labels is not None
        self._verified("head.thresholds", old.thresholds)
        bias = self._verified("head.bias", old.bias)
        weight = None if old.weight is None else self._verified("head.weight", old.weight)
        limits = updated(current, thresholds, old.labels)
        for array in (bias, weight):
            if array is not None:
                array.setflags(write=False)
        content = replace(old, weight=weight, bias=bias, thresholds=limits)
        return type(self)._new(self._encoder, content, source=self._source, note=self._note)

    @classmethod
    @validate_arguments
    def load(cls, path: PathLike, mmap: bool = False) -> Self:
        """Open a classifier saved with ``save``, complete or standalone."""
        return cls._load(path, mmap)

    @classmethod
    @validate_arguments
    def from_pretrained(
        cls,
        repo_or_path: ModelRef,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: PathLike | None = None,
        local_files_only: bool = False,
        mmap: bool = False,
    ) -> Self:
        """The classifier of a local ``.safetensors`` file or of an fse folder written by
        ``save_pretrained``, local or a Hugging Face Hub repository (``namespace/name``) at
        ``revision`` (default branch if None), checked against its commit; ``mmap`` as for
        ``load``, for local files.

        A folder or repository with the export of a Model2Vec ``StaticModelPipeline``
        (``config.json`` with ``head_config``, ``head.safetensors``, ``model.safetensors``,
        ``tokenizer.json``) is converted in memory: its table, tokenizer and one linear
        layer with softmax or sigmoid (Model2Vec's ``n_layers=0``). Hidden layers, regression
        heads, quantization, separate token weights and ``pipeline.skops`` raise; nothing is
        trained and no pickle or skops is opened. Set cutoffs of a multilabel head with
        ``with_thresholds``. A head trained with Model2Vec's default ``normalize=True`` runs
        on the full mean; one trained with ``normalize=False`` folds into the table."""
        files = _pretrained.files(
            repo_or_path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        config = _model2vec_files.claimed(files)
        if config is None:
            return cls._from_files(files, mmap)
        found, encoder = _model2vec_files.read(files, config, mmap)
        return cls._imported(found, encoder, None, "Model2Vec export")

    @classmethod
    @validate_arguments
    def from_model2vec(cls, model: Any, *, thresholds: ThresholdInput = None) -> Self:
        """The head of a finished Model2Vec classifier: a ``StaticModelPipeline`` or a
        trained ``StaticModelForClassification`` (only its ``to_pipeline`` runs), with
        its tokenizer and table, which may have been fine-tuned. Files go through
        ``from_pretrained``. Only model2vec 0.10 objects with one linear layer
        (``n_layers=0``) and the activation softmax or sigmoid are read; the refusals are
        those of ``from_pretrained``. Imports nothing of model2vec or torch.

        ``thresholds`` as for ``from_arrays``, for sigmoid (multilabel) heads only; the
        default cutoff is 0.5 as Model2Vec's ``predict``. Single-label heads predict the
        first class of largest logit, as Model2Vec; multilabel heads give a boolean
        (n, k) matrix where Model2Vec gives lists of labels (``classes_[mask]``)."""
        found, encoder = _model2vec_objects.read(model)
        return cls._imported(found, encoder, thresholds, "Model2Vec pipeline")

    @classmethod
    def _imported(cls, found: Any, encoder: Any, thresholds: Any, source: str) -> Self:
        return cls._build(
            encoder,
            found.weight,
            found.bias,
            normalized=None,
            link=found.link,
            labels=found.labels,
            source=source,
            thresholds=thresholds,
            note=found.note,
        )

    @property
    def classes_(self) -> NDArray[Any]:
        """The labels in output order (int64 or str)."""
        assert self._content.labels is not None
        return classes(self._content.labels)

    def _check_proba(self) -> None:
        if self._content.link is None:
            source = f" ({self._source})" if self._source else ""
            raise ValueError(
                f"this classifier{source} has no probabilities (link none); use decision_function"
            )
