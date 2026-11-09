"""The decision rule of binary-recall@2, fixed before the run (DR-255 plan, section 7, K)."""

from __future__ import annotations

import json
from pathlib import Path

from _quality.binary_inputs import MODELS, OVERSAMPLES_2, PROBE_QUERIES, QUERIES, VARIANTS

THRESHOLD = 0.95


def _table(path):
    """(model, store, variant, oversample) -> mean recall@10, and the queries of the records."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    table = {(r["model"], r["store"], r["variant"], r["oversample"]): r for r in rows}
    return table, {r["queries"] for r in rows}


def _meets(table, store, variant, oversample):
    return all(table[(m, store, variant, oversample)]["recall_at_10"] >= THRESHOLD for m in MODELS)


def _first(table, variant):
    """The smallest factor at which both models reach the threshold on the float32 store."""
    return next((f for f in OVERSAMPLES_2 if _meets(table, "float32", variant, f)), None)


def select(table):
    """Rule steps 2 to 5 on a complete table: the variant with the smaller factor (A on a
    tie), then that variant's smallest factor at or above it where the int8 store also
    reaches the threshold for both models."""
    first = {v: _first(table, v) for v in VARIANTS}
    found = [(f, v) for v, f in first.items() if f is not None]
    base = {"first_factor_float32": first}
    if not found:
        return {**base, "outcome": "no_variant_meets_threshold"}
    _, variant = min(found, key=lambda item: (item[0], VARIANTS.index(item[1])))
    for factor in (f for f in OVERSAMPLES_2 if f >= first[variant]):
        if _meets(table, "float32", variant, factor) and _meets(table, "int8", variant, factor):
            return {**base, "outcome": "variant_selected", "variant": variant, "oversample": factor}
    return {**base, "outcome": "no_variant_meets_threshold", "variant_tried": variant}


def _complete(table):
    return all(
        (m, s, v, f) in table
        for m in MODELS
        for s in ("float32", "int8")
        for v in VARIANTS
        for f in OVERSAMPLES_2
    )


def decide(path, cells, finished, failed):
    """failed, inconclusive, probe (a short run decides nothing) or the selection."""
    if failed:
        return {"result": "failed", "decision": None}
    status = {r["cell_id"]: r["status"] for r in finished}
    if {status.get(c["id"]) for c in cells} != {"valid"} or {c["model"] for c in cells} != set(
        MODELS
    ):
        return {"result": "inconclusive", "decision": None}
    table, queries = _table(path)
    if not _complete(table):
        return {"result": "inconclusive", "decision": None}
    if queries == {PROBE_QUERIES}:
        return {"result": "probe", "decision": None}
    if queries != {QUERIES}:  # a mix of query counts decides nothing
        return {"result": "inconclusive", "decision": None}
    summary = {
        "|".join(map(str, key)): round(row["recall_at_10"], 4) for key, row in sorted(table.items())
    }
    lower = {"|".join(map(str, key)): round(row["ci_low"], 4) for key, row in sorted(table.items())}
    return {
        "result": "complete",
        "threshold": THRESHOLD,
        "decision": select(table),
        "recall_at_10": summary,
        "bootstrap_lower_bound_reported_only": lower,
    }
