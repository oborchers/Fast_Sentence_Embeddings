#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Labels and values from stored scores (docs/architecture/scores.md, "Derived outputs").

What a head gives follows from what it stored, by the table below. All of it is a function of
the stored float32 values through the link and the label rule, computed by the functions of
the backend, so the bits are those of the run that stored them.

=========  ===========  =====================  ==============================  ========
stored     decision     proba                  labels                          values
=========  ===========  =====================  ==============================  ========
decision   the logits   link of the logits     linear@1 from the logits;       error
                                               linear@2 from the derived
                                               probabilities and the cutoffs
proba      error        the probabilities      linear@2 from the stored        error
                                               probabilities; linear@1: error
values     error        error                  error                           the values
=========  ===========  =====================  ==============================  ========

The backends take C-contiguous float32 matrices. The columns of one head of a store of
several heads are not contiguous, so each block copies them into a buffer that is reused
(at most ``block_rows`` times the width times 4 bytes); a store of a single head is passed
as it is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import Head, Link
from fse._head._content import classes
from fse._head._derive import decide, derive
from fse._store._scores_info import ScoreHead

from ._batches import ScoreBatch


class HeadPlan:
    """How one head gives its output, checked before any block is read."""

    def __init__(self, head: ScoreHead, want: str, limits: ndarray | None):
        self.head, self.want, self.limits = head, want, limits
        self.link = None if head.link == "none" else Link.parse(head.link)
        self.rule = Head.parse(head.rule)
        self.classes = None if head.labels is None else classes(head.labels)
        self.width = head.width
        _check(head, want, self.rule)

    def give(self, host: Any, view: ndarray, work: dict[str, ndarray], first_row: int) -> ndarray:
        """The output of the rows ``view`` (the columns of the head), numbered from
        ``first_row`` of the store."""
        head, want = self.head, self.want
        if head.stored == "values" or (head.stored == "decision" and want == "decision"):
            return _owned(view, flat=self._flat())
        if head.stored == "proba" and want == "proba":
            return _owned(view, flat=False)
        rows = _contiguous(view, work, head.name)
        if head.stored == "proba":  # the label rule linear@2 on the stored probabilities
            out = decide(host, rows, self.link, self.limits)
        else:
            kind = "labels" if want == "labels" else "proba"
            out = derive(
                host, rows, kind, rule=self.rule, link=self.link, thresholds=self.limits,
                first_row=first_row,
            )  # fmt: skip
        if want == "labels" and self.classes is not None and self.link is not Link.MULTILABEL:
            return self.classes[out]
        return out

    def _flat(self) -> bool:
        """Whether one column is given as (n,): always for a classifier, for a regressor of
        one target that was a vector."""
        return self.head.task == "classification" or self.head.flat is True


def _check(head: ScoreHead, want: str, rule: Head) -> None:
    """Raise unless a head that stored ``head.stored`` can give ``want``."""
    stored = head.stored
    ok = (
        (stored == "values" and want == "values")
        or (stored == "decision" and want in ("decision", "proba", "labels"))
        or (stored == "proba" and (want == "proba" or (want == "labels" and rule is Head.LINEAR2)))
    )
    if ok:
        return
    hint = f"; write the store with store={{{head.name!r}: ...}} to keep what you need"
    if stored == "proba" and want == "labels":
        reason = "its label rule linear@1 decides on the logits, which the store does not hold"
    else:
        reason = f"it cannot be derived from the stored {stored}"
    raise ValueError(f"head {head.name!r} cannot give {want!r}: {reason}{hint}")


def _owned(view: ndarray, *, flat: bool) -> ndarray:
    """A copy of the stored columns, as ``Heads`` gives them: (n,) where the head is flat."""
    if flat and view.shape[1] == 1:
        return view[:, 0].copy()
    return np.array(view, copy=True, order="C")


def _contiguous(view: ndarray, work: dict[str, ndarray], name: str) -> ndarray:
    """``view`` as a C-contiguous float32 matrix: itself, or a copy in the buffer of ``name``."""
    if view.flags.c_contiguous:
        return view
    buffer = work.get(name)
    if buffer is None or len(buffer) < len(view):
        buffer = work[name] = np.empty((len(view), view.shape[1]), np.float32)
    rows = buffer[: len(view)]
    rows[...] = view
    return rows


def predict(
    host: Any,
    plans: Mapping[str, HeadPlan],
    columns: Mapping[str, ndarray],
    store_row: int,
    work: dict[str, ndarray],
    doc_ids: ndarray | None,
) -> ScoreBatch:
    """The batch of the rows starting at ``store_row`` whose stored columns (by head) are
    ``columns``."""
    outputs = {
        name: plan.give(host, columns[name], work, store_row) for name, plan in plans.items()
    }
    n = len(next(iter(columns.values())))
    return ScoreBatch(store_row=store_row, n_rows=n, outputs=outputs, doc_ids=doc_ids)
