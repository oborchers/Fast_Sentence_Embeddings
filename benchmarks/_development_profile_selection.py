"""Select an addressed candidate search child from a sealed development comparison."""

from __future__ import annotations

import statistics

from _development_path_evidence import validate as validate_path
from _development_statistics import _ratio, order, summary
from _harness.collect import verify
from _harness.profile import identity
from _harness.records import document, read_rows


def select(root, cell_id):
    _, done = verify(root)
    if type(done["exit_status"]) is not int or done["exit_status"] != 0:
        raise ValueError("prefilter profile origin did not finish successfully")
    cell = _cell(root, cell_id)
    rows = [
        r
        for r in read_rows(root, "operations.jsonl")
        if r["cell_id"] == cell_id and r["phase"] == "comparison"
    ]
    if len(rows) != cell["pairs"] * len(order(cell, 0)) or any(
        r.get("diagnostic_profile") or r["seconds"] <= 0 for r in rows
    ):
        raise ValueError("invalid or incomplete prefilter comparison")
    _complete_pairs(cell, rows)
    factors = [(_ratio(cell, rows, n), n) for n in range(cell["pairs"])]
    summary([f for f, _ in factors])  # rejects nonfinite ratios before selecting a median
    median = statistics.median(f for f, _ in factors)
    _, number = min(factors, key=lambda item: (abs(item[0] - median), item[1]))
    row = min(
        (r for r in rows if r["pair"] == number and r["arm"] == "candidate"),
        key=lambda r: r["position"],
    )
    tag = f"{cell_id}-comparison-{number}-{row['position']}"
    validate_path(row, document(root / (tag + ".job.json")), candidate=True)
    return {
        "cell": cell_id,
        "phase": "comparison",
        "pair": number,
        "job": tag,
        "python": identity(row, "development")["executable"],
        "rule": "declared prefilter candidate; comparison pair nearest median ratio",
        "open_evidence": [],
    }


def _complete_pairs(cell, rows):
    for number in range(cell["pairs"]):
        expected = {(arm, position) for position, arm in enumerate(order(cell, number))}
        actual = [(r["arm"], r["position"]) for r in rows if r["pair"] == number]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("missing, duplicate or unknown prefilter comparison arm")


def _cell(root, cell_id):
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
    return cell


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
