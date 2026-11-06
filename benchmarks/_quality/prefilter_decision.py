"""The decision rule of the DR-168 gate, fixed before the run (plan, section 7)."""

from __future__ import annotations

import json
from pathlib import Path

THRESHOLD, REJECT = 0.10, 0.25
FULL_N, QUERIES, SETS = 859_346, 256, ("stsb-test", "wikitext-test")
MODELS = ("minishlab/potion-base-8M", "sentence-transformers/static-retrieval-mrl-en-v1")


def _medians(path):
    """model -> set -> median survivor share (scheme A, k=10, one query, 859,346 rows,
    256 queries); None for a set that is missing."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    rows = [
        r
        for r in rows
        if r.get("scope") == "main"
        and r["k"] == 10
        and r["scheme"] == "A"
        and r["queries"] == 1
        and r["n"] == FULL_N
        and r["samples"] == QUERIES
    ]
    return {
        m: {
            s: next((r["median"] for r in rows if r["model"] == m and r["set"] == s), None)
            for s in SETS
        }
        for m in MODELS
    }


def decide(path, cells, finished, failed):
    """failed, inconclusive, or the placement of DR-168 by the worse query set per model.
    A complete decision needs the probe and both models, each on the full store with
    all 256 queries of both sets."""
    if failed:
        return {"result": "failed", "decision": None}
    status = {r["cell_id"]: r["status"] for r in finished}
    scopes = sorted(c["scope"] for c in cells)
    if scopes != ["main", "main", "probe"] or {status.get(c["id"]) for c in cells} != {"valid"}:
        return {"result": "inconclusive", "decision": None}
    medians = _medians(path)
    if any(v is None for sets in medians.values() for v in sets.values()):
        return {"result": "inconclusive", "decision": None}
    worst = {m: max(sets.values()) for m, sets in medians.items()}
    return {"result": "complete", "M": worst, "medians": medians, **_place(worst)}


def _place(worst):
    if all(v <= THRESHOLD for v in worst.values()):
        return {"decision": "stays in 2.1 as a candidate; next step: class D prototype"}
    note = "moves to 2.2"
    if all(v > REJECT for v in worst.values()):
        note += (
            "; recommend not building exactly this filter (scalar int8 query, Cauchy-Schwarz bound)"
        )
    return {"decision": note}
