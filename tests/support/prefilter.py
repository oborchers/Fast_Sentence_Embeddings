"""The prefilter driver and the stores of the tests of ``fse._reference_prefilter``.

``search`` runs the phases of the rule on the reference: the queries as the exact path takes
them (``normalize@1``, then ``quantize_rows@1``), ``bounds``, one ``int8_pass`` per block with the
threshold of the blocks before it, the final ``survivors`` and the exact scores of the survivors.
Anything outside the domain returns the exact result with the reason, as the search does.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from fse import _reference as exact_path
from fse import _reference_prefilter as prefilter
from fse._contract import Normalize

F32 = np.float32


class Run(NamedTuple):
    """ids and scores of every query (as lists), the reason for a fallback or None, and the
    rows (global, ascending) that went to the exact scores per query."""

    ids: list[np.ndarray]
    scores: list[np.ndarray]
    fallback: str | None
    rows: tuple[np.ndarray, ...] | None


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def int8_store(codes):
    """int8 codes and the stored inverse norms of the contract."""
    codes = np.ascontiguousarray(codes, dtype=np.int8)
    return codes, exact_path.inverse_norms(codes)


def prepared(queries):
    """``units``, ``qcodes``, ``qscales``, ``slack`` and ``ok`` of float32 queries."""
    units = exact_path.postprocess(np.asarray(queries, dtype=F32), [Normalize()])
    qcodes, qscales = exact_path.quantize_rows(units, "int8")
    slack, ok = prefilter.bounds(units, qcodes, qscales)
    return units, qcodes, qscales, slack, ok


def exact(stored, inverse, queries, k):
    """The exact top k of every query over all rows, by the reference."""
    cosines = exact_path.cosines(stored, inverse, np.asarray(queries, dtype=F32))
    ids = [exact_path.top_k(cosines[:, j], k) for j in range(cosines.shape[1])]
    return ids, [cosines[i, j] for j, i in enumerate(ids)]


def search(stored, inverse, queries, k, *, block=4096, order=None, compact="final", run_pass=None):
    """The search with the prefilter; ``order`` permutes the blocks, ``run_pass`` replaces
    ``int8_pass``, ``compact="none"`` keeps what the passes kept without the final cut."""
    queries = np.asarray(queries, dtype=F32)
    ids, scores = exact(stored, inverse, queries, k)
    _, qcodes, qscales, slack, ok = prepared(queries)
    if not ok.all():
        return Run(ids, scores, "query_range", None)
    run_pass = run_pass or prefilter.int8_pass
    starts = list(range(0, len(stored), block))
    starts = [starts[i] for i in order] if order is not None else starts
    threshold = np.full(len(queries), -np.inf)
    heaps, parts = [], []
    for first in starts:
        part = slice(first, first + block)
        got = run_pass(stored[part], inverse[part], qcodes, qscales, slack, threshold, k)
        if got.status != "ok":
            return Run(ids, scores, "unsafe_row", None)
        heaps.append(got.heap)
        parts.append((first, got))
        threshold = prefilter.kth_estimate(prefilter.merge_heaps(heaps, k))
    rows = _candidates(parts, threshold, slack, len(queries), compact)
    return _rescored(stored, inverse, queries, rows, k)


def _candidates(parts, threshold, slack, width, compact):
    out = []
    for j in range(width):
        rows = np.concatenate([first + got.rows[j] for first, got in parts] or [[]]).astype(
            np.int64
        )
        upper = np.concatenate([got.upper[j] for _, got in parts] or [[]])
        if compact == "final":
            rows = prefilter.survivors(rows, upper, float(threshold[j]), float(slack[j]))
        out.append(np.sort(rows))
    return tuple(out)


def _rescored(stored, inverse, queries, rows, k):
    got_ids, got_scores = [], []
    for j, kept in enumerate(rows):
        scores = exact_path.cosines(stored[kept], inverse[kept], queries[j : j + 1])[:, 0]
        order = exact_path.top_k(scores, k)
        got_ids.append(kept[order])
        got_scores.append(scores[order])
    return Run(got_ids, got_scores, None, rows)


def same_result(run, ids, scores):
    """Whether ids and score bits of a run equal those of the exact path."""
    return all(
        np.array_equal(a, b) and np.array_equal(bits(s), bits(t))
        for a, b, s, t in zip(run.ids, ids, run.scores, scores, strict=True)
    )
