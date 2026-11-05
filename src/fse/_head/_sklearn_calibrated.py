#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Platt-calibrated scikit-learn classifiers, read from their attributes.

Only ``CalibratedClassifierCV(method="sigmoid", ensemble=False)`` is imported: it holds one
fitted classifier and one sigmoid per decision column, which are affine in ``x`` and fold
into the weights and bias (``fse._head._calibration``). Every other layout raises: the
private attributes are checked one by one, so a version that changes them is refused with a
clear message instead of read wrongly. Nothing of scikit-learn is imported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from fse._contract import Head

from . import _calibration

_NAME = "CalibratedClassifierCV"


def _module_name(value: Any) -> tuple[str, str]:
    cls = type(value)
    return cls.__module__, cls.__name__


def read(model: Any, decide: Callable[[Any], Any]) -> Any:
    """The head of a fitted ``CalibratedClassifierCV``: the classifier read by ``decide``
    (a ``Found`` of its decision function, the classes in ``labels``), with the sigmoid of
    every column folded in. Link ``sigmoid`` for one decision column, else ``ovr``."""
    element = _element(model)
    found = decide(element.estimator)
    labels = found.labels
    _same_classes(model.classes_, element.classes, labels)
    columns = len(found.weight)
    calibrators = element.calibrators
    if not isinstance(calibrators, list | tuple) or len(calibrators) != columns:
        count = len(calibrators) if isinstance(calibrators, list | tuple) else "no"
        raise ValueError(f"{_NAME} has {count} calibrators for {columns} decision columns")
    a, c = _sigmoids(calibrators)
    weight, bias = _calibration.platt(found.weight, found.bias, a, c)
    estimator = found.source
    text = (
        f"Calibrated with Platt scaling (scikit-learn {_NAME}, method='sigmoid', "
        f"ensemble=False, estimator {estimator}), folded into the weights and bias."
    )
    return replace(
        found,
        weight=weight,
        bias=bias,
        link="sigmoid" if columns == 1 else "ovr",
        source=f"{_NAME}({estimator})",
        flat=True,
        rule=Head.LINEAR2,
        note=text,
    )


def _element(model: Any) -> Any:
    """The one fitted calibrated classifier of ``model``, after the checks of its settings."""
    elements = getattr(model, "calibrated_classifiers_", None)
    if elements is None:
        raise ValueError(f"{_NAME} has no calibrated_classifiers_; is it fitted?")
    ensemble = getattr(model, "ensemble", True)
    if ensemble is True:
        raise ValueError(
            f"{_NAME}(ensemble=True) averages several sigmoid curves, which is not a linear "
            "head; refit with ensemble=False"
        )
    if ensemble is not False:
        raise ValueError(
            f"{_NAME}(ensemble={ensemble!r}) is not imported; refit with ensemble=False "
            "(exactly one calibrated classifier)"
        )
    method = getattr(model, "method", None)
    if method != "sigmoid":
        raise ValueError(
            f"{_NAME}(method={method!r}) cannot be folded into a linear head; only "
            "method='sigmoid' (Platt) with ensemble=False is imported"
        )
    if not isinstance(elements, list | tuple) or len(elements) != 1:
        count = len(elements) if isinstance(elements, list | tuple) else "no"
        raise ValueError(
            f"{_NAME} holds {count} calibrated classifiers; exactly one (ensemble=False) is "
            "imported"
        )
    element = elements[0]
    if _module_name(element) != ("sklearn.calibration", "_CalibratedClassifier"):
        raise ValueError(f"{_NAME} holds a {type(element).__name__}, an unknown layout")
    if getattr(element, "method", None) != "sigmoid":
        raise ValueError(f"the calibrated classifier of {_NAME} is not a sigmoid calibration")
    return element


def _same_classes(model_classes: Any, element_classes: Any, labels: Any) -> None:
    """The classes of the model, of its calibrated classifier and of the classifier under it
    must agree in value, order and type, or the columns of the sigmoids are ambiguous."""
    arrays = [np.asarray(model_classes), np.asarray(element_classes), np.asarray(labels)]
    first = arrays[0]
    for other in arrays[1:]:
        if (
            first.shape != other.shape
            or first.dtype.kind != other.dtype.kind
            or not np.array_equal(first, other)
        ):
            raise ValueError(
                f"the classes of {_NAME} and of its estimator differ: "
                f"{first.tolist()!r} and {other.tolist()!r}"
            )


def _sigmoids(calibrators: Any) -> tuple[Any, Any]:
    """``a_`` and ``b_`` of every sigmoid as float64 vectors."""
    a: list[float] = []
    c: list[float] = []
    for i, calibrator in enumerate(calibrators):
        if _module_name(calibrator) != ("sklearn.calibration", "_SigmoidCalibration"):
            name = type(calibrator).__name__
            raise ValueError(f"calibrator {i} of {_NAME} is a {name}, not a sigmoid calibration")
        for values, name in ((a, "a_"), (c, "b_")):
            number = getattr(calibrator, name, None)
            if (
                isinstance(number, bool | np.bool_)
                or not isinstance(number, float | np.floating)
                or not np.isfinite(number)
            ):
                raise ValueError(
                    f"calibrator {i} of {_NAME} has a_ or b_ that is not a finite number"
                )
            values.append(float(number))
    return np.array(a, dtype=np.float64), np.array(c, dtype=np.float64)
