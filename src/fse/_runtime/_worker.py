#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Thread-local workspace and scratch of encoding tasks."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from numpy import ndarray

from fse._text import new_workspace

from ._lengths import LengthSettings


class WorkerLocal(threading.local):
    """What one worker thread keeps during a run (its workspace and scratch), and the length
    settings of the run, which every thread sees (a plain attribute set in the calling
    thread would not reach the workers)."""

    def __init__(self, lengths: LengthSettings | None = None):
        self.lengths = lengths


def _workspace(local: threading.local) -> Any:
    """The workspace of the worker thread, made at its first task; it records the length rule
    if the run asks (``local.lengths``, set by ``infer``)."""
    workspace = getattr(local, "workspace", None)
    if workspace is None:
        lengths = getattr(local, "lengths", None)
        workspace = local.workspace = new_workspace() if lengths is None else lengths.workspace()
    return workspace


def _scratch(local: threading.local, n_rows: int, dim: int) -> ndarray:
    """The float32 rows of one task before they are stored, reused by its worker thread."""
    scratch = getattr(local, "scratch", None)
    if scratch is None or len(scratch) < n_rows or scratch.shape[1] != dim:
        scratch = local.scratch = np.empty((n_rows, dim), dtype=np.float32)
    return scratch[:n_rows]
