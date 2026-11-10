#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.ScoreBatch`` (docs/architecture/scores.md, "Reading")."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ScoreBatch:
    """One block of a score store: its rows ``store_row`` to ``store_row + n_rows - 1`` of
    the store, the ``outputs`` of every head by name and the ``doc_ids`` of the rows (None
    if the store has none).

    ``ScoreStore.iter_batches`` gives the raw stored values as read-only views of the mapped
    shards (a head of several columns is not contiguous); ``predict_batches`` gives labels or
    values as ``Heads`` does. Read the fields by name: a batch is not iterable and does not
    unpack, so a field added in a later minor release (with a default) breaks no caller."""

    store_row: int
    n_rows: int
    outputs: Mapping[str, NDArray[Any]]
    doc_ids: NDArray[np.int64] | None = None
