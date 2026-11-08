"""The versioned cells@1 decision over named main and edge cells.

This module consumes observations, never schedules or measures work. Missing,
partial and noisy evidence stays open. Only a measured failure is final early.
"""

from __future__ import annotations

import math

# These profiles run through the development controller's paired comparison.
PAIRED_PROFILES = {
    "standard@1", "encoder@1", "document-ids@1", "document-ids-1m@1", "length-report@1", "linear-heads@1", "linear-heads-multi@1",
    "linear-heads-scores@1", "int8-prefilter-search@1", "document-search-cost@1",
}
FULL_PHASES = {"all", "compare", "confirm"}


def fraction(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be a finite fraction in (0, 1]")
    return value


def validate(rule, cells):
    """Validate against all resolved cells before selection, without changing the rule."""
    fields = {"rule", "practical_saving", "main", "edge", "stop_early"}
    if not isinstance(rule, dict) or set(rule) != fields or rule["rule"] != "cells@1":
        raise ValueError("cells@1 requires exactly rule, practical_saving, main, edge, stop_early")
    if fraction(rule["practical_saving"], "practical_saving") == 1:
        raise ValueError("practical_saving must be below one")
    main, edge = rule["main"], rule["edge"]
    if not isinstance(main, list) or not main or any(not isinstance(n, str) for n in main):
        raise ValueError("main must be a nonempty list of cell names")
    if not isinstance(edge, dict) or any(not isinstance(n, str) for n in edge):
        raise ValueError("edge must map cell names to factor bounds")
    if len(set(main)) != len(main) or set(main) & set(edge):
        raise ValueError("decision roles must be disjoint and unique")
    if type(rule["stop_early"]) is not bool:
        raise ValueError("stop_early must be boolean")
    by_name = {c["id"]: c for c in cells}
    for name in [*main, *edge]:
        if name not in by_name:
            raise ValueError(f"unknown decision cell: {name}")
        _validate_role(by_name[name], name in main)
    for bound in edge.values():
        if not isinstance(bound, dict) or set(bound) != {"min_factor"}:
            raise ValueError("edge requires exactly min_factor")
        fraction(bound["min_factor"], "min_factor")


def _validate_role(cell, main):
    if cell["profile"] not in PAIRED_PROFILES:
        raise ValueError("decision roles require a paired comparison profile")
    if cell["decision_gate"] is not main:
        raise ValueError("main requires decision_gate=true; edge requires decision_gate=false")
    if cell["stop_session_on_reject"]:
        raise ValueError("decision roles cannot use stop_session_on_reject")
    if main and cell["confirmation_fraction"] < 0.5:
        raise ValueError("main requires at least half the saving in confirmation")


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"finite {name} required in decision evidence")
    return value


def _role(rule, name, row, phase):
    if row is None:
        return "missing"
    if not isinstance(row, dict) or not isinstance(row.get("status"), str):
        raise ValueError("decision evidence needs a status")
    if phase not in FULL_PHASES or row.get("functional_only"):
        return "open"
    if row.get("decision") not in ("keep", "reject", "descriptive_only"):
        return "open"
    if name in rule["main"]:
        return _main(rule, row)
    return _edge(rule["edge"][name], row)


def _main(rule, row):
    if row["decision"] == "reject":
        return "fail"
    if row["decision"] != "keep" or row["status"] != "valid":
        return "open"
    saving = number(row["comparison"]["time_saving"], "comparison saving")
    return "pass" if saving >= rule["practical_saving"] else "fail"


def _edge(bound, row):
    if row["status"] not in ("valid", "rejected"):
        return "open"
    if "comparison" not in row or "confirmation" not in row:
        return "open"
    factors = [number(row[b]["factor"]["median"], "factor") for b in ("comparison", "confirmation")]
    if min(factors) <= 0:
        raise ValueError("positive decision factors required")
    if min(factors) < bound["min_factor"]:
        return "fail"
    return "pass" if row["status"] == "valid" else "open"


def evaluate(rule, rows, selected, functional=False, phase="all", stopped_after=None):
    """Return a snapshot. final means immutable, complete means every role was decided."""
    by_name = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("cell_id"), str):
            raise ValueError("decision evidence needs a cell_id")
        name = row["cell_id"]
        if name in by_name:
            raise ValueError(f"duplicate decision evidence: {name}")
        by_name[name] = row
    roles = {
        name: _role(rule, name, by_name.get(name) if name in selected else None, phase)
        for name in [*rule["main"], *rule["edge"]]
    }
    complete = all(value in ("pass", "fail") for value in roles.values())
    decision = _overall(roles, functional)
    return {
        "rule": rule, "decision": decision,
        "final": decision in ("reject", "enable"), "complete": complete,
        "roles": roles, "stopped_after": stopped_after,
    }


def _overall(roles, functional):
    if functional:
        return "functional_only"
    if "fail" in roles.values():
        return "reject"
    if all(value == "pass" for value in roles.values()):
        return "enable"
    return "inconclusive"
