#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Calibrations folded into a head on import: Platt scaling and a softmax temperature.

Both are affine maps of the logits, so they fold into ``A`` and ``b``: the arithmetic runs
once in float64 on the backend (``affine64`` and ``quotient64``, in the floating-point
environment of the kernels) and the head is rounded once to float32 by the conversion of
the contract (``_content.weights``). Nothing of it is stored in an artifact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from fse._backends import get_backend
from fse._contract import Link

from . import _content

# Integers up to 2**53 are exact in float64: beyond, the division would round twice.
_EXACT = 2**53


def platt(weight: ndarray, bias: ndarray, a: ndarray, c: ndarray) -> tuple[ndarray, ndarray]:
    """The head of ``sigmoid(-(a z + c))`` of the logits ``z = A x + b`` of float64 ``A``
    (k, d) and ``b`` (k,), one ``a`` and ``c`` per output: ``W = (-a) A``,
    ``b' = (-a) b + (-c)``, float64. Negation is exact, so these are the bits of
    ``-(a A)`` and ``-(a b + c)``, except that a zero may have the other sign (the
    probabilities do not change)."""
    return get_backend("cpu").affine64(weight, bias, -a, -c)


def temperature(
    weight: Any, bias: Any, value: float, link: Any
) -> tuple[ndarray | Any, ndarray | Any]:
    """``weight`` and ``bias`` of a softmax head divided by the temperature ``value``, in
    float64 (rounded once to float32 by the caller). Anything else than a softmax raises.
    A shape that is not a head is returned as given, for ``_content.weights`` to refuse.
    The numbers reach the backend as they are: it converts them to float64 in the
    floating-point environment of the kernels."""
    w = _checked(weight, "weight")
    shape = np.shape(w)
    k = 1 if len(shape) == 1 else shape[0] if len(shape) == 2 else 0
    if _content.link(link, k) is not Link.SOFTMAX:
        raise ValueError(f"temperature applies only to the link softmax, not {link}")
    b = None if bias is None else _checked(bias, "bias")
    if k == 0 or not shape[-1] or (b is not None and np.shape(b) not in ((), (k,))):
        return weight, bias
    w = w.reshape(k, -1)
    b = np.zeros(k) if b is None else np.broadcast_to(b, (k,))
    return get_backend("cpu").quotient64(w, b, float(value))


def note(value: float) -> str:
    """The text of the model card for a temperature."""
    return f"Softmax temperature T = {value!r}, folded into the weights and bias."


def _checked(value: Any, name: str) -> ndarray:
    """``value`` as an array of numbers, checked by the type before any conversion: bools
    and wider floats raise, and so do integers beyond 2**53 (exact in float64 up to there),
    also among the floats of a list, where numpy would round them silently."""
    if not isinstance(value, ndarray):
        for item in np.asarray(value, dtype=object).ravel():
            _check_item(item, name)
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.dtype == np.bool_:
        raise TypeError(f"{name} must be numbers (float or int), not {array.dtype}")
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        raise TypeError(f"{name} must not be wider than float64, not {array.dtype}")
    if array.dtype.kind in "iu" and array.size:
        _check_range(int(array.max()), int(array.min()), name)
    return array


def _check_item(item: Any, name: str) -> None:
    if isinstance(item, bool | np.bool_):
        raise TypeError(f"{name} must be numbers (float or int), not bool")
    if isinstance(item, int):
        _check_range(item, item, name)


def _check_range(high: int, low: int, name: str) -> None:
    if high > _EXACT or low < -_EXACT:
        raise ValueError(f"{name} holds integers beyond 2**53, which float64 cannot hold exactly")
