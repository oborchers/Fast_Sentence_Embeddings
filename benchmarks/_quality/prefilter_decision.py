"""The decision rule of the DR-168 gate, fixed before the run (plan, section 7)."""

from __future__ import annotations

import json
from pathlib import Path

THRESHOLD, REJECT = 0.10, 0.25
MODELS = ("minishlab/potion-base-8M", "sentence-transformers/static-retrieval-mrl-en-v1")


def _medians(path):
    """model -> set -> median survivor fraction (scheme A, k=10, one query, largest N)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    rows = [
        r
        for r in rows
        if r.get("scope") == "main" and r["k"] == 10 and r["scheme"] == "A" and r["queries"] == 1
    ]
    largest = {m: max((r["n"] for r in rows if r["model"] == m), default=0) for m in MODELS}
    return {
        m: {r["set"]: r["median"] for r in rows if r["model"] == m and r["n"] == largest[m]}
        for m in MODELS
    }


def decide(path, cells, finished, failed):
    """failed, inconclusive, or the placement of DR-168 by the worse query set per model."""
    if failed:
        return {"result": "failed", "decision": None}
    status = {r["cell_id"]: r["status"] for r in finished}
    required = [status.get(c["id"]) for c in cells if c["scope"] == "main"]
    if len(required) < len(MODELS) or set(required) != {"valid"}:
        return {"result": "inconclusive", "decision": None}
    medians = _medians(path)
    worst = {m: max(sets.values()) for m, sets in medians.items() if sets}
    if len(worst) != len(MODELS):
        return {"result": "inconclusive", "decision": None}
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
