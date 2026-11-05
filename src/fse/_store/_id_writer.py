"""The optional ID data stream, published in the same directory as its vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._columns import StreamColumn
from ._id_stats import IdStats


class IdColumn(StreamColumn):
    """The document IDs of a shard: a stream column that also gathers their statistics."""

    def __init__(self, directory: Path, capacity: int):
        super().__init__(directory, "doc_ids", np.dtype("<i8"), (), capacity)
        self.stats = IdStats()

    def commit(self, rows: np.ndarray) -> None:
        super().commit(rows)
        self.stats.update(rows)
