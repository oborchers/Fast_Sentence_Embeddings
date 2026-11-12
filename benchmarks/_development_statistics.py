"""Paired ratios and the declared practical decision rule; no outlier deletion."""

from __future__ import annotations

import math
import statistics

from _development_search import enabled as search_cost
from _harness.pressure import noise, report

GATE_MINIMUMS = {"noise_multiplier": 3.0, "minimum_saving": 0.02, "confirmation_fraction": 0.5}


def decision_enabled(cell):
    return cell["decision_gate"] is True and all(
        cell[name] >= minimum for name, minimum in GATE_MINIMUMS.items()
    )


def summary(values):
    if not values or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values
    ):
        raise ValueError("positive finite raw observations required")
    q1, _, q3 = (
        statistics.quantiles(values, n=4, method="inclusive")
        if len(values) > 1
        else (values[0],) * 3
    )
    return {
        "raw": values,
        **noise(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "max_deviation": max(abs(v - 1) for v in values),
        "quartile_method": "inclusive linear interpolation",
    }


def order(cell, pair):
    if cell["order"] == "abba":
        return ("base", "candidate", "candidate", "base")
    return ("base", "candidate") if pair % 2 == 0 else ("candidate", "base")


def paired(cell, records):
    if any(row.get("diagnostic_profile") for row in records):
        raise ValueError("diagnostic profiles cannot enter paired comparisons")
    factors = [_ratio(cell, records, number) for number in range(cell["pairs"])]
    if len(records) != cell["pairs"] * len(order(cell, 0)):
        raise ValueError("unexpected extra paired observations")
    result = {
        "factor": summary(factors),
        "cpu_pressure": report(records),
        "seconds": {
            name: summary([r["seconds"] for r in records if r["arm"] == name])
            for name in ("base", "candidate")
        },
        "time_saving": 1 - 1 / statistics.median(factors),
    }
    if search_cost(cell):
        result["milliseconds_per_call"] = {
            name: summary(
                [
                    1000 * r["seconds"] / r["operation_repetitions"]
                    for r in records
                    if r["arm"] == name
                ]
            )
            for name in ("base", "candidate")
        }
    return result


def _ratio(cell, records, number):
    rows = [row for row in records if row["pair"] == number]
    expected = order(cell, number)
    if [r["arm"] for r in rows] != list(expected) or [r["position"] for r in rows] != list(
        range(len(expected))
    ):
        raise ValueError("missing, duplicate or incorrectly ordered paired arms")
    a = sum(r["seconds"] for r in rows if r["arm"] == "base")
    b = sum(r["seconds"] for r in rows if r["arm"] == "candidate")
    if a <= 0 or b <= 0:
        raise ValueError("nonpositive paired arm time")
    if search_cost(cell):
        return _search_ratio(rows, a, b)
    return a / b


def _search_ratio(rows, base_seconds, candidate_seconds):
    base_calls = sum(r["operation_repetitions"] for r in rows if r["arm"] == "base")
    candidate_calls = sum(r["operation_repetitions"] for r in rows if r["arm"] == "candidate")
    if min(base_calls, candidate_calls) <= 0:
        raise ValueError("positive search call counts required")
    return (base_seconds / base_calls) / (candidate_seconds / candidate_calls)


def decide(cell, aa, main, confirmation):
    if aa["output"] != main["output"] or main["output"] != confirmation["output"]:
        raise ValueError("complete output differs between A/A, comparison and confirmation")
    threshold = max(cell["noise_multiplier"] * aa["factor"]["iqr"], cell["minimum_saving"])
    primary = main["time_saving"] >= threshold
    confirmed = (
        confirmation["time_saving"] > 0
        and confirmation["time_saving"] >= cell["confirmation_fraction"] * main["time_saving"]
    )
    quiet = aa["factor"]["iqr"] < cell["target_aa_iqr"]
    decision = (
        "inconclusive_noise" if not quiet else ("keep" if primary and confirmed else "reject")
    )
    return {
        "decision": decision if decision_enabled(cell) else "descriptive_only",
        "threshold": threshold,
        "primary_passes": primary,
        "confirmation_passes": confirmed,
        "aa": aa,
        "comparison": main,
        "confirmation": confirmation,
        "maximum_is_gate": False,
    }
