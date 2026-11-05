#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The outputs of a head derived from its finished logits, on data alone
(docs/architecture/scores.md, "Derived outputs").

Probabilities are a function of the float32 logits through the link; the label rule
``linear@1`` decides on the logits, ``linear@2`` on the probabilities (with the cutoffs for
``multilabel@1``). The functions take the rule, the link and the cutoffs as values, so the
store mode of the heads and the reader of a score store derive their outputs in one place,
with the same bits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import Head, Link


def proba_width(k: int, link: Link | None) -> int:
    """The probabilities of a row of ``k`` logits: 2 for one logit, else ``k``; ``k`` for
    ``multilabel@1`` (one per label)."""
    if link is None:
        raise ValueError("the head has no link, so no probabilities")
    return 2 if k == 1 and link is not Link.MULTILABEL else k


def decide(
    host: Any, probabilities: ndarray, link: Link | None, thresholds: ndarray | None
) -> ndarray:
    """The labels of the label rule linear@2 from the probabilities: an indicator per label
    for ``multilabel@1`` (true strictly above the cutoff), else the index of the largest."""
    if link is Link.MULTILABEL:
        if thresholds is None:
            raise ValueError("a multilabel head needs its thresholds")
        out = np.empty(probabilities.shape, dtype=np.bool_)
        host.decisions(probabilities, thresholds, out)
        return out
    return np.argmax(probabilities, axis=1).astype(np.int64)


def derive(
    host: Any,
    logits: ndarray,
    kind: str,
    *,
    rule: Head,
    link: Link | None,
    thresholds: ndarray | None,
    first_row: int,
) -> ndarray:
    """``kind`` ("logits", "proba" or "labels") of the float32 ``logits`` of a head, which
    start at row ``first_row`` of the run (an error names the row)."""
    if kind == "logits":
        return logits
    if kind == "labels" and rule is Head.LINEAR2:
        probabilities = np.empty((len(logits), proba_width(logits.shape[1], link)), np.float32)
        host.outputs(logits, "proba", link, probabilities, first_row)
        return decide(host, probabilities, link, thresholds)
    if kind == "labels":
        out = np.empty(len(logits), dtype=np.int64)
        host.outputs(logits, "labels", None, out, first_row)
        return out
    out = np.empty((len(logits), proba_width(logits.shape[1], link)), dtype=np.float32)
    host.outputs(logits, "proba", link, out, first_row)
    return out
