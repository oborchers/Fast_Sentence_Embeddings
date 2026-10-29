#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Linear heads of fitted scikit-learn estimators, read from their attributes.

Nothing of scikit-learn is imported: an estimator is known by the name of its class in a
module of ``sklearn``, and its head by ``coef_``, ``intercept_``, ``classes_`` and the
attributes that decide its probabilities (``multi_class``, ``solver``, ``loss``), as the
installed version applies them. A ``Pipeline`` may hold an optional ``Normalizer(norm="l2")``
(the head was trained on normalized vectors) and an optional ``StandardScaler``, which folds
into A and b in float64 before the head is rounded once to float32. Anything else raises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy import ndarray


@dataclass(frozen=True)
class Found:
    """A head read from scikit-learn: A (k, d) and b (k,) in float64, the link name
    (``sigmoid``, ``softmax``, ``ovr`` or ``none``), the labels of a classifier, whether
    a ``Normalizer`` came first, and the estimator it came from."""

    weight: ndarray
    bias: ndarray
    link: str
    labels: list | None
    normalizer: bool
    source: str
    flat: bool = True  # coef_ of one dimension: predict gives (n,), not (n, 1)


def flag(found: Found, normalized: bool | None) -> bool | None:
    """The flag of the head: True after a ``Normalizer`` (False then raises), else as given
    (None: the encoder's setting)."""
    if found.normalizer:
        if normalized is False:
            raise ValueError(
                "the pipeline normalizes its input (Normalizer), so its head reads normalized "
                "vectors; normalized=False contradicts it"
            )
        return True
    return normalized


def _name(model: Any) -> str:
    cls = type(model)
    return cls.__name__ if cls.__module__.startswith("sklearn.") else ""


def _array(model: Any, name: str) -> ndarray:
    value = getattr(model, name, None)
    if value is None:
        raise ValueError(f"{type(model).__name__} has no {name}; is it fitted?")
    if not isinstance(value, ndarray | float | int | np.floating | np.integer):
        raise ValueError(
            f"{type(model).__name__}.{name} is a {type(value).__name__}, not a dense array "
            "(densify a sparse coef_ first)"
        )
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{type(model).__name__}.{name} holds values that are not finite")
    return array


def _linear(model: Any) -> tuple[ndarray, ndarray]:
    """coef_ as (k, d) and intercept_ as (k,)."""
    weight = _array(model, "coef_")
    weight = weight[None, :] if weight.ndim == 1 else weight
    if weight.ndim != 2:
        raise ValueError(f"coef_ must be a matrix, not of shape {weight.shape}")
    bias = np.broadcast_to(_array(model, "intercept_").reshape(-1), (len(weight),)).copy()
    features = getattr(model, "n_features_in_", weight.shape[1])
    if features != weight.shape[1]:
        raise ValueError(f"coef_ has {weight.shape[1]} columns for {features} features")
    return weight, bias


def _classes(model: Any, k: int) -> list:
    classes = getattr(model, "classes_", None)
    if not isinstance(classes, ndarray) or classes.ndim != 1:
        raise ValueError(
            f"{type(model).__name__}.classes_ must be one array of labels (multi-label "
            "classifiers are not supported)"
        )
    want = 2 if k == 1 else k
    if len(classes) != want:
        raise ValueError(f"{len(classes)} classes for {k} rows of coef_")
    return classes.tolist()


def installed_version() -> tuple[int, int]:
    """(major, minor) of the installed scikit-learn, from its distribution's metadata
    (nothing of scikit-learn is imported)."""
    from importlib.metadata import version

    major, minor = version("scikit-learn").split(".")[:2]
    return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)


def _logistic_link(model: Any, k: int) -> str:
    """The rule of the installed version: from 1.8 sigmoid for two classes and softmax
    else; before, ovr where ``multi_class`` says so, or under auto for two classes, for the
    solver liblinear, and before 1.6 for newton-cholesky (1.6 gave it the multinomial loss,
    scikit-learn PR 28840, and predict_proba softmax), else softmax, which for two classes
    is softmax of [-z, z]."""
    multi_class = getattr(model, "multi_class", None)
    if multi_class is None:  # scikit-learn 1.8 and newer
        return "sigmoid" if k == 1 else "softmax"
    binary = k == 1
    ovr_solvers = (
        ("liblinear", "newton-cholesky") if installed_version() < (1, 6) else ("liblinear",)
    )
    ovr = multi_class in ("ovr", "warn") or (
        multi_class in ("auto", "deprecated") and (binary or model.solver in ovr_solvers)
    )
    if ovr:
        return "sigmoid" if binary else "ovr"
    return "softmax2" if binary else "softmax"


def _sgd_link(model: Any, k: int) -> str:
    loss = getattr(model, "loss", None)
    if loss in ("log_loss", "log"):
        return "sigmoid" if k == 1 else "ovr"
    if loss == "modified_huber":
        raise ValueError(
            "SGDClassifier(loss='modified_huber') has probabilities fse does not compute; "
            "train with loss='log_loss', or use another loss for decisions only"
        )
    return "none"


_CLASSIFIERS: dict[str, Callable[[Any, int], str]] = {
    "LogisticRegression": _logistic_link,
    "LogisticRegressionCV": _logistic_link,
    "SGDClassifier": _sgd_link,
    **dict.fromkeys(
        (
            "LinearSVC",
            "RidgeClassifier",
            "RidgeClassifierCV",
            "Perceptron",
            "PassiveAggressiveClassifier",
        ),
        lambda _model, _k: "none",
    ),
}
_REGRESSORS = frozenset(
    {
        "LinearRegression",
        "Ridge",
        "RidgeCV",
        "Lasso",
        "LassoCV",
        "ElasticNet",
        "ElasticNetCV",
        "SGDRegressor",
        "LinearSVR",
        "HuberRegressor",
    }
)


def _estimator(model: Any, task: str) -> Found:
    name = _name(model)
    known = _CLASSIFIERS if task == "classification" else _REGRESSORS
    if name not in known:
        kinds = ", ".join(sorted(known))
        raise ValueError(
            f"{type(model).__module__}.{type(model).__name__} is not a scikit-learn {task} "
            f"head fse can read; supported: {kinds}, alone or in a Pipeline"
        )
    weight, bias = _linear(model)
    if task == "regression":
        flat = np.ndim(getattr(model, "coef_", None)) == 1
        return Found(weight, bias, "none", None, False, name, flat)
    labels = _classes(model, len(weight))
    link = _CLASSIFIERS[name](model, len(weight))
    if link == "softmax2":  # a binary multinomial model: one logit z, softmax of [-z, z]
        link = "softmax"
    return Found(weight, bias, link, labels, False, name)


def _scaled(found: Found, scaler: Any) -> Found:
    """``A ((x - mean) / scale) + b`` as ``(A / scale) x + (b - (A / scale) mean)``, with
    the steps the scaler applies: it centers only with ``with_mean`` and scales only with
    ``with_std`` (``mean_`` is kept for the variance even without centering)."""
    weight, bias = found.weight, found.bias
    scale = getattr(scaler, "scale_", None) if getattr(scaler, "with_std", True) else None
    mean = getattr(scaler, "mean_", None) if getattr(scaler, "with_mean", True) else None
    if scale is not None:
        scale = _array(scaler, "scale_")
        if scale.shape != (weight.shape[1],):
            raise ValueError(
                f"the StandardScaler has {scale.size} features, the head {weight.shape[1]}"
            )
        weight = weight / scale
    if mean is not None:
        mean = _array(scaler, "mean_")
        if mean.shape != (weight.shape[1],):
            raise ValueError(
                f"the StandardScaler has {mean.size} features, the head {weight.shape[1]}"
            )
        bias = bias - weight @ mean
    return replace(found, weight=weight, bias=bias)


def _pipeline_steps(model: Any) -> list:
    steps = getattr(model, "steps", None)
    if not isinstance(steps, list) or not steps:
        raise ValueError("the Pipeline has no steps")
    return [
        est
        for _, est in steps
        if est is not None and not (isinstance(est, str) and est == "passthrough")
    ]


def read(model: Any, task: str) -> Found:
    """The head of a fitted estimator, or of a Pipeline of an optional ``Normalizer``, an
    optional ``StandardScaler`` and the estimator."""
    if _name(model) != "Pipeline":
        return _estimator(model, task)
    *before, last = _pipeline_steps(model)
    found = _estimator(last, task)
    normalizer = bool(before) and _name(before[0]) == "Normalizer"
    if normalizer:
        if getattr(before[0], "norm", None) != "l2":
            raise ValueError(
                f"a Normalizer before the head must use norm='l2', not {before[0].norm!r}"
            )
        before = before[1:]
    if before and _name(before[0]) == "StandardScaler":
        found = _scaled(found, before[0])
        before = before[1:]
    if before:
        names = ", ".join(type(step).__name__ for step in before)
        raise ValueError(
            f"the Pipeline holds {names}; only an optional Normalizer(norm='l2') and an "
            "optional StandardScaler may come before the head, in this order"
        )
    return replace(found, normalizer=normalizer)
