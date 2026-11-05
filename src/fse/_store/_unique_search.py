"""Search with at most one hit per document: the k best documents, each by its best row.

The tasks are those of the filtered and the plain search (a block of rows against a block
of queries, scored by the backend or BLAS), but a task keeps the best row of every
document that can still be among the best ``k`` of a query (``_unique_docs``), and the
floor of a query is the score of its k-th best document. The scores are those of the
ordinary search, so ``mode="exact"`` gives its bits for the rows it returns.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from fse import _log, _reference
from fse._contract import Normalize

from . import _filtered, _search, _selection
from ._unique_docs import DocumentCandidates, best_per_doc


def _blocks(shards: Any, n_queries: int, decode: int, values: Any, kind: str):
    """(shard index, local positions): the blocks of the plain search, or those of the
    selection; the positions are strictly ascending."""
    if values is None:
        for index, begin, end, _ in _search.blocks(shards, n_queries, decode):
            yield index, np.arange(begin, end, dtype=np.int64)
        return
    size = max(1, min(65_536, (1 << 20) // min(n_queries, 256)))
    if shards:  # bound the gathers and the BLAS decoding, also for very wide exact rows
        size = min(size, max(1, (16 << 20) // (4 * shards[0][1]["vectors"].shape[1])))
    yield from _selection.blocks(shards, values, kind, size)


def _tasks(shards: Any, n_queries: int, decode: int, values: Any, kind: str, count: list[int]):
    for index, positions in _blocks(shards, n_queries, decode, values, kind):
        shard, tensors = shards[index]
        stored = _filtered._part(tensors["vectors"], positions)
        inverse = _filtered._part(tensors["inv_norms"], positions)
        docs = _filtered._part(tensors["doc_ids"], positions)
        physical = positions + shard.first_row
        count[0] += len(positions)
        for qb in _search.query_blocks(n_queries):
            yield stored, inverse, docs, shard.id_stats.sorted, physical, qb


def search(
    backend: Any,
    shards: Any,
    queries: np.ndarray,
    k: int,
    values: Any,
    kind: str,
    *,
    workers: int,
    exact: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rows (int64), scores (float32) and document IDs (int64) of the ``k`` best
    documents of every query, each (queries, min(k, documents)); ``values`` and ``kind``
    select the rows as for the filtered search, or are None for all rows."""
    _search.check_queries(backend, queries)
    normalized = None if exact else _reference.postprocess(queries, [Normalize()])
    candidates = DocumentCandidates(len(queries), k)
    count = [0]

    def run(task):
        stored, inverse, docs, ordered, physical, qb = task
        floor = candidates.floor[qb]
        try:
            scores = (
                backend.cosines(stored, inverse, queries[qb])
                if normalized is None
                else _search._blas_cosines(stored, inverse, normalized[qb])
            )
        except ValueError as error:
            message = re.sub(r"\brow (\d+)\b", lambda m: f"row {physical[int(m[1])]}", str(error))
            raise ValueError(f"queries {qb.start} to {qb.stop - 1}: {message}") from None
        q, scores, local, labels = best_per_doc(scores, docs, ordered, k, floor)
        return qb.start, (q, scores, physical[local], labels)

    decode = 0 if exact else queries.shape[1]
    tasks = _tasks(shards, len(queries), decode, values, kind, count)
    for result in _search._run(run, tasks, workers, "fse-search"):
        candidates.add(*result)
    _log.emit(
        "store.search",
        "fse store: %d queries over %d rows, best row per document",
        len(queries),
        count[0],
        queries=len(queries),
        k=k,
        mode="exact" if exact else "blas",
        rows=count[0],
        workers=workers,
        max_per_doc=1,
    )
    return candidates.result()
