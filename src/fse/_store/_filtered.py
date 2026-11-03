"""Filtered exact/BLAS scoring with physical positions retained through top-k."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from fse import _log, _reference
from fse._contract import Normalize

from . import _search, _selection


def _part(tensor: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if positions[-1] - positions[0] + 1 == len(positions):
        return tensor[positions[0] : positions[-1] + 1]
    return tensor[positions]


def _tasks(shards: Any, values: np.ndarray, kind: str, queries: int, count: list[int]):
    size = max(1, min(65_536, (1 << 20) // min(queries, 256)))
    if shards:
        # Bound gathers as well as BLAS decoding, including very wide exact rows.
        width = shards[0][1]["vectors"].shape[1]
        size = min(size, max(1, (16 << 20) // (4 * width)))
    for index, positions in _selection.blocks(shards, values, kind, size):
        shard, tensors = shards[index]
        stored = _part(tensors["vectors"], positions)
        inverse = _part(tensors["inv_norms"], positions)
        physical = positions + shard.first_row
        count[0] += len(positions)
        for qb in _search.query_blocks(queries):
            yield stored, inverse, physical, qb


def search(
    backend: Any,
    shards: Any,
    queries: np.ndarray,
    k: int,
    values: np.ndarray,
    kind: str,
    *,
    workers: int,
    exact: bool,
) -> tuple[np.ndarray, np.ndarray]:
    _search.check_queries(backend, queries)
    normalized = None if exact else _reference.postprocess(queries, [Normalize()])
    candidates = _search.Candidates(len(queries), k)
    count = [0]

    def run(task):
        stored, inverse, physical, qb = task
        try:
            scores = (
                backend.cosines(stored, inverse, queries[qb])
                if normalized is None
                else _search._blas_cosines(stored, inverse, normalized[qb])
            )
        except ValueError as error:
            message = re.sub(r"\brow (\d+)\b", lambda m: f"row {physical[int(m[1])]}", str(error))
            raise ValueError(f"queries {qb.start} to {qb.stop - 1}: {message}") from None
        q, scores, local = _search.best(scores, 0, k, candidates.floor[qb])
        return qb.start, (q, scores, physical[local])

    for result in _search._run(
        run, _tasks(shards, values, kind, len(queries), count), workers, "fse-search"
    ):
        candidates.add(*result)
    _log.emit(
        "store.search",
        "fse store: %d queries over %d selected rows",
        len(queries),
        count[0],
        queries=len(queries),
        k=k,
        mode="exact" if exact else "blas",
        rows=count[0],
        workers=workers,
    )
    return candidates.result(min(k, count[0]))


def document_ids(shards: Any, rows: np.ndarray) -> np.ndarray:
    """Copy final labels while the caller holds the store's active-use guard."""
    out = np.empty(rows.shape, dtype=np.int64)
    for shard, tensors in shards:
        mask = (rows >= shard.first_row) & (rows < shard.first_row + shard.rows)
        out[mask] = tensors["row_ids"][rows[mask] - shard.first_row]
    return out
