"""Reject short fse calls individually, without hiding them behind a median."""

import math


def qualifies(row):
    minimum = row.get("minimum_call_seconds", 0)
    if minimum == 0:
        return True
    values = [row.get("first_encode_s", row.get("first_predict_s")), *row.get("warm_seconds", [])]
    return (
        math.isfinite(minimum)
        and minimum > 0
        and len(values) >= 2
        and all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= minimum
            for value in values
        )
    )
