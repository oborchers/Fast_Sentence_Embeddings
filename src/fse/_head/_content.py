#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What a linear head is: its task, A and b, the labels, the link and its flag.

``HeadContent`` holds a checked head; ``weights``, ``labels`` and ``link`` check what a
caller gives (``from_arrays``) or a file holds, each raising a clear error before anything
is computed. Weights and biases are converted to float32 by the conversion of the
contract (round to nearest, in the floating-point environment of the kernels), and
must be finite after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from fse._backends import get_backend
from fse._contract import LINK_NONE, Link

TASKS = ("classification", "regression")
# The links a caller names, and what "auto" takes for k outputs.
LINK_NAMES = {"sigmoid": Link.SIGMOID, "softmax": Link.SOFTMAX, "ovr": Link.OVR, "none": None}
_INT64 = (-(2**63), 2**63 - 1)


@dataclass(frozen=True, eq=False)
class HeadContent:
    """A checked head: ``weight`` (k, d) float32 (None for a standalone head, whose table
    is folded), ``bias`` (k,), whether it was trained on ``normalized`` rows, its ``link``
    (None: no probabilities) and, for classification, its ``labels`` in output order (two
    for k = 1, negative first, else k)."""

    task: str
    weight: ndarray | None
    bias: ndarray
    normalized: bool
    link: Link | None
    labels: tuple | None
    standalone: bool = False
    # One output gives (n,) rather than (n, 1): always for classification (as scikit-learn's
    # decision_function), for regression when the head was one vector (coef_ of shape (d,)).
    flat: bool = True

    @property
    def k(self) -> int:
        return len(self.bias)

    @property
    def link_id(self) -> str:
        return LINK_NONE if self.link is None else self.link.id


def weights(weight: Any, bias: Any, dim: int) -> tuple[ndarray, ndarray]:
    """A (k, d) or (d,) and b (k,), a scalar or None (zeros) as read-only float32 arrays of
    a head on rows of ``dim`` values."""
    w = _numbers(weight, "weight")
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2 or w.shape[1] != dim or not len(w):
        raise ValueError(
            f"weight must have shape (k, {dim}) or ({dim},) for rows of {dim} values, not "
            f"{np.shape(weight)}"
        )
    k = len(w)
    if bias is None:
        b = np.zeros(k, dtype=np.float32)
    else:
        b = _numbers(bias, "bias")
        if b.ndim == 0:
            b = np.full(k, b, dtype=np.float32)
        if b.shape != (k,):
            raise ValueError(
                f"bias must hold {k} values (one per output) or one, not {np.shape(bias)}"
            )
    _finite(w, "weight")
    _finite(b, "bias")
    for array in (w, b):
        array.setflags(write=False)
    return w, b


def _numbers(value: Any, name: str) -> ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.dtype == np.bool_:
        raise TypeError(f"{name} must be numbers (float or int), not {array.dtype}")
    if array.ndim > 2:
        raise ValueError(f"{name} must have at most two dimensions, not {array.ndim}")
    # ascontiguousarray makes a scalar one value of one dimension: keep the shape given.
    return np.array(get_backend("cpu").float32(array), copy=True).reshape(array.shape)


def _finite(array: ndarray, name: str) -> None:
    bad = np.argwhere(~np.isfinite(array))
    if len(bad):
        where = ", ".join(str(int(i)) for i in bad[0])
        raise ValueError(f"{name}[{where}] is not finite in float32")


def labels(values: Any, k: int) -> tuple:
    """The labels of a classifier with k outputs: two for k = 1 (negative first), else k;
    all int or all str, unique."""
    if isinstance(values, str | bytes) or not hasattr(values, "__len__"):
        raise TypeError(f"labels must be a sequence of int or str, not {type(values).__name__}")
    items = [_label(v) for v in (values.tolist() if isinstance(values, ndarray) else values)]
    want = 2 if k == 1 else k
    if len(items) != want:
        raise ValueError(
            f"a head of {k} output{'s' * (k != 1)} needs {want} labels"
            f"{' (negative first)' if k == 1 else ''}, not {len(items)}"
        )
    if len({type(v) for v in items}) > 1:
        raise TypeError("labels must be all int or all str, not a mix")
    if len(set(items)) != len(items):
        raise ValueError(f"labels must be unique, not {items}")
    return tuple(items)


def _label(value: Any) -> int | str:
    if isinstance(value, bool | np.bool_):
        raise TypeError("a label must be an int or a str, not a bool")
    if isinstance(value, int | np.integer):
        value = int(value)
        if not _INT64[0] <= value <= _INT64[1]:
            raise ValueError(f"the label {value} is beyond int64")
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("a label must have a valid UTF-8 representation") from error
        return str(value)
    raise TypeError(f"a label must be an int or a str, not {type(value).__name__}")


def link(name: Any, k: int) -> Link | None:
    """The link a classifier of k outputs takes: ``"auto"`` is sigmoid for k = 1 and
    softmax else; ``"sigmoid"`` needs k = 1, ``"ovr"`` k >= 2, ``"softmax"`` of one output z
    is softmax of [-z, z] (a binary multinomial model of scikit-learn); ``"none"`` gives
    no probabilities."""
    if name == "auto":
        return Link.SIGMOID if k == 1 else Link.SOFTMAX
    if not isinstance(name, str) or name not in LINK_NAMES:
        raise ValueError(f"link must be one of auto, {', '.join(LINK_NAMES)}, not {name!r}")
    chosen = LINK_NAMES[name]
    if chosen is Link.SIGMOID and k != 1:
        raise ValueError(f"the link sigmoid takes one output, not {k}; use softmax or ovr")
    if chosen is Link.OVR and k == 1:
        raise ValueError("the link ovr takes at least two outputs; use sigmoid for one")
    return chosen


def classes(values: tuple) -> ndarray:
    """The labels as numpy array: int64, or str."""
    if isinstance(values[0], str):
        # Fixed-width numpy strings discard terminal NULs when read as scalars.
        dtype = object if any(value.endswith("\0") for value in values) else np.str_
        return np.array(values, dtype=dtype)
    return np.array(values, dtype=np.int64)
