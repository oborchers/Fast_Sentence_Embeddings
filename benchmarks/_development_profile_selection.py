"""Select an addressed candidate search child from a sealed development comparison."""

from __future__ import annotations

import statistics

from _harness.collect import verify
from _harness.profile import identity
from _harness.records import document, read_rows


def select(root, cell_id):
    _, done = verify(root)
    if type(done["exit_status"]) is not int or done["exit_status"] != 0:
        raise ValueError("prefilter profile origin did not finish successfully")
    environment = document(root / "environment.json")
    cells = [c for c in environment["contract"]["cells"] if c["id"] == cell_id]
    if len(cells) != 1:
        raise ValueError("exactly one declared prefilter profile cell required")
    cell = cells[0]
    completed = [r for r in read_rows(root, "cells.jsonl") if r["cell_id"] == cell_id]
    if len(completed) != 1 or completed[0]["status"] != "valid":
        raise ValueError("prefilter profile cell is missing or invalid")
    if (
        cell["profile"] != "int8-prefilter-search@1"
        or cell["stage"] != "store-search"
        or cell["output_dtype"] != "int8"
    ):
        raise ValueError("profile origin must be an int8 prefilter search cell")
    rows = [
        r
        for r in read_rows(root, "operations.jsonl")
        if r["cell_id"] == cell_id and r["phase"] == "comparison"
    ]
    pairs = {}
    for row in rows:
        key = (row["pair"], row["arm"])
        if key in pairs or row["arm"] not in ("base", "candidate"):
            raise ValueError("duplicate or unknown prefilter profile arm")
        if row.get("diagnostic_profile") or row["seconds"] <= 0:
            raise ValueError("invalid prefilter profile operation")
        pairs[key] = row
    expected = {(n, arm) for n in range(cell["pairs"]) for arm in ("base", "candidate")}
    if pairs.keys() != expected:
        raise ValueError("missing prefilter comparison pairs")
    factors = [
        (pairs[n, "base"]["seconds"] / pairs[n, "candidate"]["seconds"], n)
        for n in range(cell["pairs"])
    ]
    median = statistics.median(f for f, _ in factors)
    _, number = min(factors, key=lambda item: (abs(item[0] - median), item[1]))
    row = pairs[number, "candidate"]
    return {
        "cell": cell_id,
        "phase": "comparison",
        "pair": number,
        "job": f"{cell_id}-comparison-{number}-{row['position']}",
        "python": identity(row, "development")["executable"],
        "rule": "declared prefilter candidate; comparison pair nearest median ratio",
        "open_evidence": [],
    }


def profile_spec(args, selected, budget):
    from _allocation_profile import profile_spec as speed_spec

    return {
        **speed_spec(args, selected, budget),
        "entry": "development",
        "source_results": str(args.allocation_root / "prefilter-results"),
        "nominal_arm": "candidate",
        "action": "measure",
        "python": selected["python"],
        "datasets": None,
    }
