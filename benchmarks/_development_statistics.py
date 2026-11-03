"""Paired ratios and the declared practical decision rule; no outlier deletion."""

from __future__ import annotations

import math
import statistics


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
    factors = []
    for pair in range(cell["pairs"]):
        rows = [row for row in records if row["pair"] == pair]
        expected = order(cell, pair)
        if [r["arm"] for r in rows] != list(expected) or [r["position"] for r in rows] != list(
            range(len(expected))
        ):
            raise ValueError("missing, duplicate or incorrectly ordered paired arms")
        a = sum(r["seconds"] for r in rows if r["arm"] == "base")
        b = sum(r["seconds"] for r in rows if r["arm"] == "candidate")
        if a <= 0 or b <= 0:
            raise ValueError("nonpositive paired arm time")
        factors.append(a / b)
    if len(records) != cell["pairs"] * len(order(cell, 0)):
        raise ValueError("unexpected extra paired observations")
    return {
        "factor": summary(factors),
        "seconds": {
            name: summary([r["seconds"] for r in records if r["arm"] == name])
            for name in ("base", "candidate")
        },
        "time_saving": 1 - 1 / statistics.median(factors),
    }


def decide(cell, aa, main, confirmation):
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
        "decision": decision if cell["decision_gate"] else "descriptive_only",
        "threshold": threshold,
        "primary_passes": primary,
        "confirmation_passes": confirmed,
        "aa": aa,
        "comparison": main,
        "confirmation": confirmation,
        "maximum_is_gate": False,
    }
