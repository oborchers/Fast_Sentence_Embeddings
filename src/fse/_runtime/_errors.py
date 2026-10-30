#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The error of a task: that of its earliest failing row, whatever its boundaries.

The stages of a task run one after the other over all its rows, so the first error a
task meets is the first of its first failing stage, not of its first failing row: a
later row that overflows in pooling would hide an earlier row that cannot be
normalized. After a task failed, ``earliest`` runs it again on shorter prefixes, by
bisection, until it has the shortest prefix that fails. Every row is a pure function of
its sentence, so that prefix fails exactly at its last row, in the first stage that row
fails in, and its error is the one the run reports (ARCHITECTURE.md, "Runtime",
Errors). This costs nothing until a task fails, and then about log2 of its rows runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from numpy import ndarray


def earliest(run: Callable[[int], None], rows: int) -> None:
    """``run(rows)``; if it raises a ValueError (every error of a row is one), raise the
    error of the shortest failing prefix ``run(k)`` instead. Other errors pass as they
    are."""
    try:
        run(rows)
        return
    except ValueError as error:
        failed = error
    good, bad = 0, rows  # run(good) succeeds (nothing to run for 0), run(bad) fails
    while bad - good > 1:
        middle = (good + bad) // 2
        try:
            run(middle)
        except ValueError as error:
            bad, failed = middle, error
        else:
            good = middle
    raise failed


def first_rows(target: Any, rows: int) -> Any:
    """The first ``rows`` rows of a target: an array, a tuple of arrays (None kept), or
    None."""
    if target is None:
        return None
    if isinstance(target, ndarray):
        return target[:rows]
    return tuple(None if part is None else part[:rows] for part in target)
