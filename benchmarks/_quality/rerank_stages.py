"""First stages of the reranker study: dense search, BM25, Reciprocal Rank Fusion, reranking.

A ranking is an int64 matrix of document positions, one row per query, best first, ``-1``
after the end. Ties always break by the position in the corpus, so a ranking is a function of
the scores alone.
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np

RRF_K = 60  # Cormack, Clarke and Buettcher (SIGIR 2009), fixed before the study
DEPTH = 1000


def _order(scores, excluded, depth):
    """The best ``depth`` positions by score, ties by position; ``excluded`` never appears."""
    scores = np.asarray(scores, dtype=np.float64).copy()
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite first-stage score")
    if excluded is not None:
        scores[excluded] = -np.inf
    order = np.lexsort((np.arange(len(scores)), -scores))
    order = order[np.isfinite(scores[order])][:depth]
    row = np.full(depth, -1, np.int64)
    row[: len(order)] = order
    return row


def _excluded(data, query):
    """The position of the document whose ID equals the query ID, where the task ignores it."""
    if not data.ignore_identical_ids:
        return None
    hit = [i for i, d in enumerate(data.doc_ids) if d == data.query_ids[query]]
    return hit or None


def _rank_all(score_rows, data, depth):
    return np.stack([_order(row, _excluded(data, q), depth) for q, row in enumerate(score_rows)])


def dense(encoder, data, depth=DEPTH):
    """Exact cosine search with the normalized vectors of an encoder (an MTEB adapter)."""
    docs = np.asarray(encoder.encode([{"text": data.docs}]), dtype=np.float32)
    queries = np.asarray(encoder.encode([{"text": data.queries}]), dtype=np.float32)
    for name, rows in (("documents", docs), ("queries", queries)):
        norms = np.linalg.norm(rows, axis=1)
        if not np.isfinite(rows).all() or np.abs(norms[norms > 0] - 1).max() > 1e-3:
            raise ValueError(f"dense {name} are not finite unit vectors")
    scores = queries @ docs.T
    return _rank_all(scores, data, depth), scores


def bm25(data, depth=DEPTH):
    """BM25 of bm25s (Lucene variant, k1 1.5, b 0.75), English stop words and Snowball stemmer."""
    import bm25s
    import Stemmer

    if version("bm25s") != "0.3.16" or version("PyStemmer") != "3.1.0":
        raise ValueError("BM25 packages differ from the pinned study")
    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    tokens = bm25s.tokenize(data.docs, stopwords="en", stemmer=stemmer, show_progress=False)
    retriever.index(tokens, show_progress=False)
    known = set(retriever.vocab_dict)
    rows = []
    for query in data.queries:
        words = bm25s.tokenize(
            [query], stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False
        )[0]
        words = [w for w in words if w in known]
        rows.append(retriever.get_scores(words) if words else np.zeros(len(data.docs)))
    return _rank_all(rows, data, depth)


def rrf(rankings, depth=DEPTH):
    """Reciprocal Rank Fusion: sum of 1 / (60 + rank) over the lists, rank counted from one.
    Ties break by the best rank in any list, then by position."""
    fused = []
    for rows in zip(*rankings, strict=True):
        score, best = {}, {}
        for row in rows:
            for rank, position in enumerate(row[row >= 0].tolist(), start=1):
                score[position] = score.get(position, 0.0) + 1.0 / (RRF_K + rank)
                best[position] = min(best.get(position, rank), rank)
        order = [p for _, _, p in sorted((-s, best[p], p) for p, s in score.items())][:depth]
        row = np.full(depth, -1, np.int64)
        row[: len(order)] = order
        fused.append(row)
    return np.stack(fused)


def rerank(ranking, depth, scores):
    """The first ``depth`` candidates ordered by the reranker score ``scores[(query, position)]``,
    ties by their first-stage rank, followed by the rest of the first-stage list."""
    result = ranking.copy()
    for q, row in enumerate(ranking):
        head = [p for p in row[:depth].tolist() if p >= 0]
        keyed = sorted((-scores[(q, p)], rank, p) for rank, p in enumerate(head))
        result[q, : len(head)] = [p for _, _, p in keyed]
    return result


def candidates(ranking, depth):
    """The (query, position) pairs among the first ``depth`` candidates."""
    return {(q, int(p)) for q, row in enumerate(ranking) for p in row[:depth] if p >= 0}
