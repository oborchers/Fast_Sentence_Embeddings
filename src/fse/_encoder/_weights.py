#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The word weights of SIF and uSIF and the component weights of uSIF, as fse 1.1.0.

Every rounding point is explicit, so the weights are the bits of 1.1.0 (numpy 2) on
every numpy version: the word probabilities are divided in float64 and rounded once, SIF
computes in float32, uSIF in float64 with one rounding at the end.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

_F32 = np.float32
_F64 = np.float64
_EPS = float(np.finfo(_F32).eps)  # the EPS of 1.x


def check_counts(counts: object, n_words: int) -> ndarray:
    """One count per word: int64, not negative, with a sum above 0 that fits int64."""
    if not isinstance(counts, ndarray) or counts.dtype.kind not in "iu" or counts.ndim != 1:
        raise TypeError("counts must be a vector of integers, one per word")
    if len(counts) != n_words:
        raise ValueError(f"counts must hold one count per word ({n_words}), not {len(counts)}")
    if not len(counts) or counts.min() < 0:
        raise ValueError("counts must not be negative")
    if int(counts.max()) > np.iinfo(np.int64).max // len(counts):
        raise ValueError("counts are too large: their sum must fit into int64")
    counts = counts.astype(np.int64)
    if not counts.sum():
        raise ValueError("all counts are zero; SIF and uSIF need word frequencies")
    return counts


def word_probabilities(counts: ndarray) -> ndarray:
    """count / sum as float32: the float32 counts divided by the sum in float64, rounded
    once (1.1.0 divided a float32 array in place by the numpy int64 sum)."""
    total = _F64(counts.sum())
    return (counts.astype(_F32).astype(_F64) / total).astype(_F32)


def sif_weights(counts: ndarray, alpha: float) -> ndarray:
    """alpha / (alpha + p(w)) in float32."""
    pw = word_probabilities(counts)
    a = _F32(alpha)
    return a / (a + pw)


def usif_weights(counts: ndarray, length: int) -> ndarray:
    """a / (a / 2 + p(w)) with a from the vocabulary size and the sentence length."""
    pw = word_probabilities(counts)
    v = len(counts)
    threshold = 1 - (1 - (1 / v)) ** length
    alpha = int(np.count_nonzero(pw > _F32(threshold))) / v
    a = (1 - alpha) / (alpha * (v / 2) + _EPS)
    return (a / (a / 2 + pw.astype(_F64))).astype(_F32)


def usif_component_weights(singular_values: ndarray) -> ndarray:
    """lambda_j^2 / sum lambda^2 of the removed components, in float32."""
    squares = singular_values.astype(_F32) * singular_values.astype(_F32)
    total = squares.sum(dtype=_F32)
    if not total > 0:
        raise ValueError("the sentence vectors have no variance; uSIF removes no component")
    return squares / total
