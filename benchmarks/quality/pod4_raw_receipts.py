"""Reconstruct every planned cell from its own raw operations and correctness receipt."""

import json

from pod_duration import qualifies
from pod_folded_summary import summarize as classify
from pod_readme_summary import summarize as encode


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def duration_rows(rows):
    return bool(rows) and all(
        row.get("minimum_call_seconds") == 2 and qualifies(row)
        for row in rows
        if row.get("runtime_package") == "fse"
    )


def readme_cell(raw, checks, cell):
    done = [r for r in raw if r.get("event") == "series_complete" and r["cell"] == cell]
    checked = [r for r in checks if r["cell"] == cell]
    if len(done) != 1 or len(checked) != 1:
        return {"cell": cell, "eligible": False}
    rows = [r for r in raw if r.get("series_id") == done[0]["series_id"]]
    operations = [r for r in rows if r.get("event") == "operation"]
    rounds = [r for r in rows if r.get("event") == "round_complete"]
    against = cell.rsplit("/", 1)[-1]
    labels = ["fse-a", "fse-b", *([] if against == "table" else [against])]
    summary = encode(done[0], operations, rounds, labels, checked[0])
    summary["eligible"] &= duration_rows(operations) and duration_rows(
        list(checked[0]["records"].values())
    )
    return summary


def readme(path, expected):
    raw = read_jsonl(path / "speed-results.jsonl")
    checks = read_jsonl(path / "output-checks.jsonl")
    return {"cells": [readme_cell(raw, checks, cell) for cell in expected]}


def classification(path):
    raw = read_jsonl(path / "speed-results.jsonl")
    result = classify(raw)
    for cell in result["cells"]:
        operations = [
            r
            for r in raw
            if r.get("event") == "operation" and r.get("series_id") == cell["series_id"]
        ]
        cell["eligible"] &= duration_rows(operations)
    return result
