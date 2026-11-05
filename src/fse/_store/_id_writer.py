"""The optional ID data stream, published in the same directory as its vectors."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fse._container import Hasher
from fse._container._npy_inplace import create, shrink, sync_data, write_at

from ._id_stats import IdStats


class IdColumn:
    """A positional file and an ordered digest; IO completion is owned by ShardWriter."""

    def __init__(self, directory: Path, capacity: int):
        self.path, self.capacity = directory / "doc_ids.npy", capacity
        self.fd, self.offset = -1, 0
        self.digest, self.stats = Hasher(), IdStats()

    def write(self, begin: int, ids: np.ndarray) -> None:
        if self.fd < 0:
            self.fd, self.offset = create(self.path, np.dtype("<i8"), (self.capacity,))
        write_at(self.fd, memoryview(ids).cast("B"), self.offset + 8 * begin)

    def commit(self, ids: np.ndarray) -> None:
        self.digest.update(ids)
        self.stats.update(ids)

    def finish(self, rows: int, *, synced: bool) -> tuple:
        if self.fd < 0:
            raise RuntimeError("doc_ids were not written; the shard is not published")
        if rows != self.capacity:
            shrink(self.fd, self.offset, np.dtype("<i8"), (rows,))
        if synced:
            sync_data(self.fd)
        self.close()
        return "I64", (rows,), self.digest.checksum()

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
