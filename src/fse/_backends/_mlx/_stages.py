#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The steps of the mlx backend on the device's rows, one stage per step on the host.

Each step runs alone on the host (``CpuBackend.steps`` with a Prepared of that step), so
that the check before it sees the rows it gets: before a linear step the rows whose
values could overflow, before ``normalize`` the rows that could be zero, tiny or too
large on either side (``_bounds``). Such rows are zeroed for the remaining stages (a zero
row passes every step) and taken from the host by the backend; the bound E of the other
rows is carried through every step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import Project, Step

from .._base import PoolInput, Prepared
from . import _bounds


@dataclass(frozen=True, eq=False)
class Stage:
    """One step: the host's Prepared of it alone, its bound (None for normalize) and the
    dimension of its input rows."""

    prepared: Prepared
    linear: _bounds.Linear | None
    dim: int


def stages(host: Any, steps: Sequence[Step], dim: int) -> tuple[Stage, ...]:
    """The stages of ``steps`` on rows of ``dim`` values, prepared on ``host``."""
    out = []
    for step in steps:
        rows = PoolInput(np.zeros((1, dim), np.float32), np.ones(1, np.float32))
        prepared = host.prepare(rows, (step,))
        out.append(Stage(prepared, _bounds.linear(step, dim), dim))
        dim = prepared.output_dim
    return tuple(out)


def run(
    host: Any,
    stages: Sequence[Stage],
    rows: ndarray,
    out: ndarray,
    first_row: int,
    error: ndarray,
    nonempty: ndarray,
    bad: ndarray,
) -> ndarray:
    """Run ``stages`` on the device rows ``rows`` (which may be ``out``) into ``out``; the
    rows to take from the host, ``bad`` and those the checks found."""
    current = rows
    for i, stage in enumerate(stages):
        g = _bounds.norms(current)
        if stage.linear is None:
            bad = bad | (_bounds.unnormalizable(g, g, error) & nonempty)
        else:
            bad = bad | stage.linear.overflows(error, g)
        current[bad] = 0
        last = i == len(stages) - 1
        target = out if last else current
        if not last and isinstance(stage.prepared.steps[0], Project):  # never in place
            target = np.empty((len(current), stage.prepared.output_dim), np.float32)
        host.steps(current, stage.prepared, first_row, target)
        if stage.linear is None:
            error = _bounds.after_normalize(error, g, stage.dim)
        else:
            error = stage.linear.propagate(error, g)
        current = target
    return bad
