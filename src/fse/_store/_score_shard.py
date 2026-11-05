#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""One shard of a score store being written (docs/architecture/scores.md, "Writing").

The shard of ``ShardBase`` (``_shard``) for the kind ``prediction_scores``: the one
streamed tensor ``scores`` of all heads, optionally the document IDs, and one constant
tensor ``thresholds.<head>`` for every multilabel head, written whole in every shard.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy import ndarray

from ._columns import ConstantColumn, StreamColumn
from ._id_writer import IdColumn
from ._scores_format import shard_metadata
from ._scores_info import ScoreInfo
from ._shard import ShardBase


class ScoreShardWriter(ShardBase):
    """A shard of scores: ``scores``, the document IDs if the store has them and the
    thresholds of the multilabel heads."""

    def __init__(
        self,
        store: Path,
        info: ScoreInfo,
        thresholds: Mapping[str, ndarray],
        first_row: int,
        capacity: int,
        created_by: str,
        *,
        synced: bool = True,
    ):
        if set(thresholds) != set(info.threshold_names):
            raise ValueError("the thresholds must be those of the multilabel heads of the store")
        super().__init__(store, first_row, capacity, created_by, synced=synced)
        self.info = info
        # scores.npy is created at the first write, so the shards reserved ahead of their
        # writes hold no open file.
        self.scores = StreamColumn(
            self.tmp, "scores", np.dtype(np.float32), (info.dim,), capacity, always=True
        )
        self.ids = IdColumn(self.tmp, capacity) if info.has_doc_ids else None
        constants = [ConstantColumn(self.tmp, n, thresholds[n]) for n in info.threshold_names]
        self.columns = [c for c in (self.scores, self.ids, *constants) if c is not None]

    def reserve(self, n_rows: int) -> tuple[ndarray]:
        """The buffer of the next ``n_rows`` (all heads side by side), which the workers fill."""
        begin = self._reserved_rows(n_rows)
        rows = np.empty((n_rows, self.info.dim), np.float32)
        self._open[begin] = rows
        self.reserved = begin + n_rows
        return (rows,)

    def commit(self, begin: int, rows: ndarray, doc_ids: ndarray | None = None) -> None:
        """The rows from ``begin`` (in the shard), which must be the buffer reserved there,
        are finished: hashed in order."""
        self._check_commit(begin, rows)
        if self.ids is not None:
            if doc_ids is None or len(doc_ids) != len(rows):
                raise ValueError("missing document IDs for committed rows")
            self.ids.commit(doc_ids)
        self.scores.commit(rows)
        self._committed_rows(begin, len(rows))

    def write(self, begin: int, rows: ndarray, doc_ids: ndarray | None = None) -> None:
        """Write the committed rows from ``begin`` (in the shard) into the files; positional,
        so it may run in another thread than ``commit`` (only one thread writes)."""
        self.scores.write(begin, rows)
        if self.ids is not None:
            if doc_ids is None:
                raise ValueError("missing document IDs for written rows")
            self.ids.write(begin, doc_ids)
        self._written_rows_done(len(rows))

    def _metadata(self, rows: int) -> dict[str, str]:
        metadata = shard_metadata(
            self.info, self.first_row, rows, self.created_by, unsynced=not self.synced
        )
        if self.ids is not None:
            metadata.update(self.ids.stats.metadata())
        return metadata
