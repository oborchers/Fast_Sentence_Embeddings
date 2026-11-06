"""Count the rows the prefilter keeps for blocks of queries and check V1 to V4 on the way."""

from __future__ import annotations

import numpy as np
from _development_budget import check

from _quality import prefilter as bound

QUERY_BLOCK = 32
GROUP = 8
SCHEMES = ("A", "A_unslacked", "B")


def count(blocks, rows, queries, ks, public=None, deadline=None):
    """Per-query rows and the counts of groups of eight queries, for every k.

    ``blocks`` is a callable giving the row blocks again for every block of queries;
    ``public`` is the (ids, scores) of the public exact search, or None to take the
    reference top-k of the scanned cosines (the synthetic probe).
    """
    if len(queries) % GROUP:
        raise ValueError("the number of queries must be a multiple of eight")
    rows_out, groups = [], {k: {"union": [], "pair": []} for k in ks}
    worst, checked = 0.0, 0
    for begin in range(0, len(queries), QUERY_BLOCK):
        check(deadline)
        block = queries[begin : begin + QUERY_BLOCK]
        found = bound.scan(blocks(), rows, block)
        worst = max(worst, found.worst)
        lists = _lists(found, public, begin, len(block), max(ks))
        masks = _block(found, begin, lists, ks, rows_out)
        _groups(masks, rows, groups)
        checked += len(block) * rows
    return rows_out, groups, {"pairs": checked, "violations": 0, "worst_over_slack": worst}


def _lists(found, public, begin, width, top):
    if public is None:
        return bound.reference_lists(found.c, top)
    ids, scores = public
    return ids[begin : begin + width], scores[begin : begin + width]


def _block(found, begin, lists, ks, rows_out):
    width = found.c.shape[1]
    masks = {k: np.zeros((width, len(found.c)), bool) for k in ks}
    for j in range(width):
        column = np.ascontiguousarray(found.c[:, j])
        estimate = np.ascontiguousarray(found.a[:, j])
        for k in ks:
            kept, kept_b = bound.survivors(
                estimate, found.bounds.slack[j], k, bound.kth_value(column, k)
            )
            bound.check_selection(column, kept, lists[0][j], lists[1][j], k)
            masks[k][j] = kept
            rows_out.append(_row(found, j, begin, k, estimate, kept, kept_b))
    return masks


def _row(found, j, begin, k, estimate, kept, kept_b):
    b = found.bounds
    return {
        "query": begin + j,
        "k": k,
        "rows": len(estimate),
        "kept_A": int(kept.sum()),
        "kept_A_unslacked": bound.unslacked(estimate, b.eps[j], k),
        "kept_B": int(kept_b.sum()),
        "scale": float(b.scale[j]),
        "max_abs": float(b.max_abs[j]),
        "eps": float(b.eps[j]),
        "slack": float(b.slack[j]),
        "slack_over_eps": float(b.slack[j] / b.eps[j]) if b.eps[j] else None,
    }


def _groups(masks, rows, groups):
    for k, mask in masks.items():
        for start in range(0, len(mask), GROUP):
            part = mask[start : start + GROUP]
            groups[k]["union"].append(int(np.logical_or.reduce(part).sum()) / rows)
            groups[k]["pair"].append(int(part.sum()) / (GROUP * rows))


def spread(values):
    values = np.asarray(values, dtype=np.float64)
    low, middle, high = np.percentile(values, [10, 50, 90])
    return {
        "median": float(middle),
        "p10": float(low),
        "p90": float(high),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "n": len(values),
    }


def summarize(rows_out, groups, rows, context):
    """One record per (k, scheme, single query), and per k for groups of eight."""
    records = []
    for k in sorted({r["k"] for r in rows_out}):
        part = [r for r in rows_out if r["k"] == k]
        for scheme in SCHEMES:
            fractions = [r["kept_" + scheme] / rows for r in part]
            proof = "not a proof" if scheme == "A_unslacked" else "provable"
            records.append(
                {**context, "k": k, "scheme": scheme, "queries": 1, "proof": proof}
                | spread(fractions)
            )
        for name, values in groups[k].items():
            records.append(
                {**context, "k": k, "scheme": "A", "queries": GROUP, "measure": name}
                | spread(values)
            )
    return records
