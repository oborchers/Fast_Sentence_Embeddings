"""Validate the full quality comparison before any card or report consumes it."""

from __future__ import annotations

import math

from hub_classic_evidence import benchmark_identity, validate_score
from hub_classic_sources import BASELINES, DTYPES, SOURCES


def repo_name(name, method, dtype):
    return f"fse/{name}-{method}" + ("" if dtype == "float32" else f"-{dtype}")


def verification_rows(verification, name):
    if verification.get("source") != name:
        raise ValueError("verification source mismatch")
    rows = verification.get("rows")
    if not isinstance(rows, list) or len(rows) != 7:
        raise ValueError("verification needs six production rows and one historical row")
    keyed = {(row.get("method"), row.get("dtype")): row for row in rows}
    expected = {(method, dtype) for method in ("raw", "sif10") for dtype in DTYPES}
    expected.add(("sif10-historical-test-fit", "float32"))
    if len(keyed) != 7 or set(keyed) != expected:
        raise ValueError("verification rows are missing, duplicated or unexpected")
    for key, row in keyed.items():
        _check_row(row, name, *key)
    for method in ("raw", "sif10"):
        _check_parent_differences(keyed, method)
    return keyed


def _check_row(row, name, method, dtype):
    expected = {"benchmark": benchmark_identity(), "pairs": 1379}
    if method == "sif10-historical-test-fit":
        expected.update(source=f"fse/{name}", source_revision=SOURCES[name])
    baseline = BASELINES[name][method != "raw"]
    expected["baseline_1x"] = baseline
    pearson = validate_score(row, expected, "pearson", -100, 100)
    _difference(row, "difference_1x_points", pearson - baseline)
    if method == "sif10-historical-test-fit":
        return
    if row.get("repo") != repo_name(name, method, dtype) or row.get("retrieval_rows") != 2758:
        raise ValueError("verification repository or retrieval row count mismatch")
    validate_score(row, {}, "default_tokenizer_pearson", -100, 100)
    validate_score(row, {}, "recall_at_10", 0, 1)


def _difference(row, key, expected):
    value = validate_score(row, {}, key, -200, 200)
    if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{key} does not match the validated Pearson scores")


def _check_parent_differences(keyed, method):
    parent = keyed[(method, "float32")]
    if parent["recall_at_10"] != 1.0:
        raise ValueError("float32 self-reference recall must be one")
    for dtype in DTYPES:
        row = keyed[(method, dtype)]
        _difference(row, "difference_float32_points", row["pearson"] - parent["pearson"])
