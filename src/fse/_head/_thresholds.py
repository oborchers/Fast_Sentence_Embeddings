"""Strict multilabel cutoffs, resolved once in the classifier's class order."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy import ndarray

from ._content import _label, _numbers


def thresholds(value: Any, labels: tuple) -> ndarray:
    """Read default, scalar, vector or partial named cutoffs as owned read-only F32."""
    if value is None:
        values = [0.5] * len(labels)
    elif isinstance(value, Mapping):
        values = _named(value, labels)
    elif np.isscalar(value):
        values = [_cutoff(value)] * len(labels)
    else:
        array = np.asarray(value)
        if array.ndim != 1 or len(array) != len(labels):
            raise ValueError(f"thresholds must hold {len(labels)} values, not shape {array.shape}")
        values = [_cutoff(item) for item in value]
    result = _numbers(values, "thresholds")
    result.setflags(write=False)
    return result


def _named(given: Mapping, labels: tuple) -> list:
    indices = {label: index for index, label in enumerate(labels)}
    values = [0.5] * len(labels)
    for key, value in given.items():
        name = _label(key)
        if name not in indices or type(name) is not type(labels[0]):
            raise ValueError(f"unknown threshold label {key!r}; expected one of {labels!r}")
        values[indices[name]] = _cutoff(value)
    return values


def _cutoff(value: Any) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise TypeError("a threshold must be a number, not a bool or string")
    if not 0 <= value <= 1:
        raise ValueError("a threshold must be finite and in [0, 1]")
    return float(value)
