"""Stores of chosen rows and an independent oracle for the tests of ``VectorStore.rescore``."""

import numpy as np

from fse import _reference
from support.take_store import write_rows

F32 = np.float32
DIM = 5
INVERSE = F32(0.25)  # what ``write_rows`` stores as the inverse norm of every row


def bits(x):
    return np.asarray(x, F32).view(np.uint32)


def make(path, dtype, n=60, *, shard_rows=7, docs=None, seed=0):
    """A store of ``n`` rows of random stored values; returns the codes and the document IDs."""
    rng = np.random.default_rng(seed)
    if dtype == "int8":
        codes = rng.integers(-127, 128, (n, DIM)).astype(np.int8)
        scales = (rng.random(n) + 0.5).astype(F32)
    else:
        codes, scales = rng.standard_normal((n, DIM)).astype(dtype), None
    codes[n // 2 :: 5] = codes[0]  # equal rows: ties by position
    write_rows(path, dtype, codes, scales, shard_rows=shard_rows, doc_ids=docs)
    return codes


def queries(n, seed=1):
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(F32)


def candidate_lists(n_rows, n_queries, width=30, seed=2):
    """A matrix of candidates: unsorted, duplicates, gaps in every place, one list of gaps."""
    rng = np.random.default_rng(seed)
    out = rng.integers(-1, n_rows, (n_queries, width)).astype(np.int64)
    out[0, :] = -1
    out[1 % n_queries, 0] = -1
    out[2 % n_queries, -1] = -1
    return out


def expected(codes, qs, cands, k, docs=None):
    """Per query: (positions, scores, document IDs) of its best rows, from the reference."""
    inverse = np.full(len(codes), INVERSE, F32)
    out = []
    for q, row in enumerate(np.asarray(cands).reshape(len(qs), -1)):
        pos = np.unique(row[row != -1])
        cos = _reference.cosines(codes[pos], inverse[pos], qs[q : q + 1])[:, 0]
        if docs is None:
            top = _reference.top_k(cos, k)
        else:
            top = _reference.top_k_unique(cos, docs[pos], k)
        out.append((pos[top], cos[top], None if docs is None else docs[pos][top]))
    return out


def assert_same(got, want, *, single=False):
    """``got`` holds ``want`` for every query, filled up on the right as documented."""
    if single:
        ((ids, scores, labels),) = want
        np.testing.assert_array_equal(got.ids, ids)
        np.testing.assert_array_equal(bits(got.scores), bits(scores))
        if labels is not None:
            np.testing.assert_array_equal(got.doc_ids, labels)
        return
    width = max(len(w[0]) for w in want)
    assert got.ids.shape == got.scores.shape == (len(want), width)
    for q, (ids, scores, labels) in enumerate(want):
        t = len(ids)
        np.testing.assert_array_equal(got.ids[q, :t], ids)
        np.testing.assert_array_equal(bits(got.scores[q, :t]), bits(scores))
        assert (got.ids[q, t:] == -1).all()
        assert (got.scores[q, t:] == -np.inf).all()
        if labels is not None:
            np.testing.assert_array_equal(got.doc_ids[q, :t], labels)
            assert (got.doc_ids[q, t:] == -1).all()
