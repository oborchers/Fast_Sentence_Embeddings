"""Reject short fse calls individually, without hiding them behind a median."""

import math


def qualifies(row):
    minimum = row.get("minimum_call_seconds", 0)
    if minimum == 0:
        return True
    values = [row.get("first_encode_s", row.get("first_predict_s")), *row.get("warm_seconds", [])]
    if row.get("timed_passes", len(values) - 1) != len(values) - 1:
        return False
    if len(values) < 2:
        return False
    nested = row.get("encoder_call_seconds")
    if row.get("kind") == "encode-matmul" and nested is None:
        return False
    if nested is not None:
        if len(nested) != len(values):
            return False
        values.extend(nested)
    return (
        math.isfinite(minimum)
        and minimum > 0
        and len(values) >= 2
        and all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= minimum
            for value in values
        )
    )
