"""The bound of a provable int8 prefilter and the checks V1 to V4 (DR-168 gate, class K).

A row survives when even its best case reaches the k-th lower bound. Nothing here times
anything or computes a kernel: the integer products are exact float64 BLAS products.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fse import _reference
from fse._backends._cpu import CpuBackend
from fse._contract import Normalize

SLACK_RATIO = 1.003  # the proof needs 1.0002 times eps (docs: plan, section 3)
SLACK_ABSOLUTE = 2.0**-14  # the proof needs 2^-16.9; d <= 1024, codes in [-127, 127]
MAX_DIM = 1024
BLOCK_ROWS = 16_384
SAMPLE_ROWS = 256


class BoundViolationError(AssertionError):
    """A check of the bound, the candidate sets or the provenance failed."""


@dataclass
class Bounds:
    """Per query: the int8 query (float64), its scale, the error norm and the slack."""

    codes: np.ndarray  # (queries, dim) float64, integers in [-127, 127]
    scale: np.ndarray  # (queries,) float64
    eps: np.ndarray  # (queries,) float64
    slack: np.ndarray  # (queries,) float64
    max_abs: np.ndarray  # (queries,) float64, the largest unit component


def validate_queries(queries):
    """Raise a ValueError before any work for a zero, non-finite or unnormalizable query."""
    queries = np.asarray(queries)
    if queries.dtype != np.float32 or queries.ndim != 2 or not len(queries):
        raise ValueError("queries must be a nonempty float32 matrix")
    if not np.isfinite(queries).all():
        raise ValueError("a query is not finite")
    if queries.shape[1] > MAX_DIM:
        raise ValueError("the bound is proven for at most 1024 dimensions")
    # The backend raises for a zero query or one that normalize@1 cannot normalize.
    CpuBackend().cosines(np.zeros((0, queries.shape[1]), np.int8), np.zeros(0, np.float32), queries)
    return queries


def query_bounds(queries):
    queries = validate_queries(queries)
    unit = _reference.postprocess(queries, [Normalize()])
    codes, scales = _reference.quantize_rows(unit, "int8")
    if codes.dtype != np.int8 or codes.shape != unit.shape or scales.dtype != np.float32:
        raise BoundViolationError("quantize_rows returned an unexpected form")
    qi, scale, exact = codes.astype(np.float64), scales.astype(np.float64), unit.astype(np.float64)
    error = exact - scale[:, None] * qi
    eps = np.sqrt((error * error).sum(axis=1))
    return Bounds(qi, scale, eps, SLACK_RATIO * eps + SLACK_ABSOLUTE, np.abs(exact).max(axis=1))


def row_blocks(vectors, inverse, size=BLOCK_ROWS):
    """(first row, int8 codes, float32 inverse norms) of consecutive blocks of one array."""
    for begin in range(0, len(vectors), size):
        yield begin, vectors[begin : begin + size], inverse[begin : begin + size]


def store_blocks(store, size=BLOCK_ROWS):
    for shard, tensors in store._shards:
        for begin, codes, inverse in row_blocks(tensors["vectors"], tensors["inv_norms"], size):
            yield shard.first_row + begin, codes, inverse


class Scan:
    """The estimates ``a`` (float64) and the exact cosines ``c`` (float32) of one block of
    queries against every row, rows first, plus the statistics of check V1."""

    def __init__(self, rows, bounds):
        width = len(bounds.eps)
        self.bounds = bounds
        self.a = np.empty((rows, width), np.float64)
        self.c = np.empty((rows, width), np.float32)
        self.violations = 0
        self.worst = 0.0  # max |c - a| / slack

    def add(self, first, codes, inverse, queries, backend):
        stop = first + len(codes)
        if codes.min() < -127:
            raise BoundViolationError("a stored code is -128 or below")
        products = codes.astype(np.float64) @ self.bounds.codes.T
        _check_integers(codes, products, self.bounds.codes)
        a = (products * self.bounds.scale) * inverse.astype(np.float64)[:, None]
        c = backend.cosines(codes, inverse, queries)
        self.a[first:stop], self.c[first:stop] = a, c
        slack = self.bounds.slack
        gap = np.abs(c.astype(np.float64) - a)
        self.violations += int((gap > slack).sum())
        self.worst = max(self.worst, float((gap / slack).max()))


def _check_integers(codes, products, query_codes):
    """The float64 products are integer sums below 2^24: compare a sample with int64."""
    head = slice(0, SAMPLE_ROWS)
    exact = codes[head].astype(np.int64) @ query_codes.astype(np.int64).T
    if not np.array_equal(products[head], exact.astype(np.float64)):
        raise BoundViolationError("the float64 products differ from the int64 sums")


def scan(blocks, rows, queries, backend=None):
    backend = backend or CpuBackend()
    result = Scan(rows, query_bounds(queries))
    for first, codes, inverse in blocks:
        result.add(first, codes, inverse, queries, backend)
    if result.violations:
        raise BoundViolationError(f"V1: {result.violations} pairs outside [L, U]")
    return result


def survivors(estimate, slack, k, exact_kth):
    """Masks of the rows kept for one query: scheme A (the k-th lower bound), scheme A
    without the slack (not a proof) and scheme B (the exact k-th value as threshold)."""
    rows = len(estimate)
    low, high = estimate - slack, estimate + slack
    threshold = np.partition(low, rows - k)[rows - k]
    kept = high >= threshold
    return kept, high >= exact_kth


def unslacked(estimate, eps, k):
    rows = len(estimate)
    threshold = np.partition(estimate - eps, rows - k)[rows - k]
    return int(np.count_nonzero(estimate + eps >= threshold))


def kth_value(column, k):
    """The k-th largest exact cosine as float64."""
    return float(np.partition(column, len(column) - k)[len(column) - k])


def check_selection(column, kept, ids, scores, k):
    """V2 and V3 for one query: the reference top-k lies in the kept rows, and the top-k
    over the kept rows gives the same ids and score bits."""
    ids, scores = np.asarray(ids)[:k], np.asarray(scores)[:k]
    if not kept[ids].all():
        raise BoundViolationError("V2: a top-k row was dropped by the prefilter")
    positions = np.flatnonzero(kept)  # ascending, so ties go to the smaller position
    order = _reference.top_k(column[positions], k)
    if not np.array_equal(positions[order], ids):
        raise BoundViolationError("V3: the top-k over the kept rows differs")
    if not np.array_equal(column[positions[order]].view(np.uint32), scores.view(np.uint32)):
        raise BoundViolationError("V3: the scores over the kept rows differ in bits")


def reference_lists(c, width):
    """The top-``width`` ids and scores of every column of ``c`` by the reference order."""
    ids = np.stack([_reference.top_k(c[:, j], width) for j in range(c.shape[1])])
    return ids, np.take_along_axis(c.T, ids, axis=1)


def provenance(blocks, rows, queries, backend=None):
    """V4: stored inverse norms equal the reference, the codes lie in [-127, 127], and
    the backend equals the reference on a sample."""
    backend = backend or CpuBackend()
    seen, sample = 0, None
    for first, codes, inverse in blocks:
        seen += len(codes)
        if codes.min() < -127:
            raise BoundViolationError("V4: a stored code is -128 or below")
        expected = _reference.inverse_norms(codes, first)
        if not np.array_equal(expected.view(np.uint32), np.asarray(inverse).view(np.uint32)):
            raise BoundViolationError("V4: stored inverse norms differ from the reference")
        if sample is None:
            sample = (np.array(codes[:2048]), np.array(inverse[:2048]))
    if seen != rows:
        raise BoundViolationError("V4: the blocks do not cover the store")
    _backend_sample(sample, queries[:4], backend)
    return {"rows": seen, "sample_rows": len(sample[0]), "sample_queries": min(4, len(queries))}


def _backend_sample(sample, queries, backend):
    codes, inverse = sample
    got = backend.cosines(codes, inverse, queries)
    want = _reference.cosines(codes, inverse, queries)
    if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
        raise BoundViolationError("V4: the backend differs from the numpy reference")
