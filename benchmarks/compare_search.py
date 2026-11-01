"""Common-vector search backends, exact ID recall and boundary-tie recall.

All indexes receive the same finite float32 matrix. FAISS uses exhaustive
IndexFlatIP. hnswlib uses HNSW (M=16, ef_construction=200, ef=100, seed=100).
Queries include their own stored row. Recall uses every query, with a float64
cosine oracle, stable lower-ID ties, and a separate boundary-equivalent score.
This is numerical index recall, not labeled retrieval relevance.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


def timed(function):
    start = time.perf_counter()
    value = function()
    return time.perf_counter() - start, value


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    from fse._container._common import _sync_directory

    _sync_directory(path.parent)


def make_index(kind: str, matrix: np.ndarray, encoder, path: Path, workers: int):
    if kind == "fse":
        return fse_index(matrix, encoder, path)
    if kind == "faiss":
        import faiss

        faiss.omp_set_num_threads(workers)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index
    import hnswlib

    index = hnswlib.Index(space="cosine", dim=matrix.shape[1])
    index.init_index(max_elements=len(matrix), ef_construction=200, M=16, random_seed=100)
    index.add_items(matrix, np.arange(len(matrix)), num_threads=workers)
    index.set_ef(100)
    index.set_num_threads(workers)
    return index


def fse_index(matrix: np.ndarray, encoder, path: Path):
    from fse import ShardSink
    from fse._backends import get_backend

    with ShardSink(path, encoder, dtype="float32", durability="full") as sink:
        target = sink.target(0, len(matrix))
        target[0][...] = matrix
        target[1][...] = get_backend("cpu").inverse_norms(matrix)
        sink.commit(0, target)
    return None


def save_index(kind: str, index, path: Path) -> None:
    if kind == "fse":
        # ShardSink already flushed and synced the complete store.
        return
    if kind == "faiss":
        import faiss

        faiss.write_index(index, str(path))
    else:
        index.save_index(str(path))
    fsync_file(path)


def open_index(kind: str, path: Path, dim: int, workers: int):
    if kind == "fse":
        from fse import VectorStore

        return VectorStore.open(path)
    if kind == "faiss":
        import faiss

        return faiss.read_index(str(path))
    import hnswlib

    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(path))
    index.set_ef(100)  # hnswlib does not persist this search setting.
    index.set_num_threads(workers)
    return index


def search(kind: str, index, queries: np.ndarray, k: int, workers: int):
    if kind.startswith("fse"):
        result = index.search(queries, k=k, workers=workers, mode="auto")
        return np.asarray(result.ids)
    if kind == "faiss":
        return index.search(queries, k)[1]
    return index.knn_query(queries, k=k, num_threads=workers)[0]


def search_all(kind: str, index, matrix: np.ndarray, k: int, workers: int, batch: int):
    durations, results = [], []
    for begin in range(0, len(matrix), batch):
        seconds, ids = timed(lambda b=begin: search(kind, index, matrix[b : b + batch], k, workers))
        durations.append(seconds)
        results.append(ids)
    return np.concatenate(results), durations


def index_recall(matrix: np.ndarray, ids: np.ndarray, k: int) -> dict:
    from compare_numerics import normalized

    vectors, nonzero = normalized(matrix)
    if ids.shape != (len(matrix), k) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("index must return an integer ID matrix of shape (queries, k)")
    if np.any(ids < 0) or np.any(ids >= len(matrix)):
        raise ValueError("index returned an ID outside the corpus")
    if any(len(set(row)) != k for row in ids):
        raise ValueError("index returned duplicate IDs for one query")
    strict = equivalent = count = boundary_ties = 0
    for begin in range(0, len(matrix), 64):
        scores = vectors[begin : begin + 64] @ vectors.T
        canonical = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        for offset, expected in enumerate(canonical):
            row = begin + offset
            if not nonzero[row]:
                continue
            threshold = scores[offset, expected[-1]]
            chosen = ids[row]
            strict += len(set(expected) & set(chosen))
            better = int(np.count_nonzero(scores[offset] > threshold))
            better_hits = int(np.count_nonzero(scores[offset, chosen] > threshold))
            boundary_hits = int(np.count_nonzero(scores[offset, chosen] == threshold))
            equivalent += better_hits + min(boundary_hits, k - better)
            boundary_ties += int(np.count_nonzero(scores[offset] == threshold) > 1)
            count += 1
    return {
        "recall_at_k": strict / (count * k) if count else None,
        "boundary_equivalent_recall_at_k": equivalent / (count * k) if count else None,
        "query_rows": count,
        "zero_queries_excluded_from_recall": int((~nonzero).sum()),
        "queries_with_boundary_ties": boundary_ties,
        "oracle": "float64 cosine over all rows, lower-ID ties, self included",
    }
