"""Whole-pool search blocks, common sizing and round summaries. Standard library only."""

from __future__ import annotations

import hashlib
import math
import statistics


def validate(cell):
    for key in ("texts", "dimensions", "workers", "queries_per_call", "pool_size", "k", "rounds"):
        if type(cell[key]) is not int or cell[key] <= 0:
            raise ValueError(f"search requires positive integer {key}")
    if cell["rounds"] < 3 or cell["pool_size"] % cell["queries_per_call"]:
        raise ValueError("search requires at least three rounds and whole query calls")
    if cell["store_dtype"] not in ("float32", "int8") or cell["mode"] not in ("exact", "binary"):
        raise ValueError("unsupported search store dtype or mode")
    if cell["prefilter"] not in ("off", "shipped"):
        raise ValueError("search prefilter must be off or shipped")
    if cell["mode"] == "binary" and (
        cell["store_dtype"] != "float32"
        or type(cell.get("oversample")) is not int
        or cell["oversample"] <= 0
        or cell["prefilter"] != "off"
    ):
        raise ValueError("binary search requires float32, positive oversample and prefilter off")
    if cell["prefilter"] == "shipped" and (
        cell["store_dtype"] != "int8"
        or cell["mode"] != "exact"
        or not 1 <= cell["queries_per_call"] <= 8
        or cell["k"] > 100
        or not 256 <= cell["dimensions"] <= 1024
        or cell["texts"] < 65536
    ):
        raise ValueError("prefilter cell is outside the supported domain")


def common_passes(calibrations, target=3.0):
    if not calibrations or any(not math.isfinite(s) or s <= 0 for s in calibrations):
        raise ValueError("positive finite pool calibration durations required")
    return max(math.ceil(target / seconds) for seconds in calibrations)


def order(cells, number):
    offset = number % len(cells)
    return cells[offset:] + cells[:offset]


def sequence_digest(query_digest, pool_size, passes, queries_per_call):
    return hashlib.sha256(
        f"search-sequence@1:{query_digest}:{pool_size}:{passes}:{queries_per_call}".encode()
    ).hexdigest()


def summarize(rows, pool_size, passes):
    values = [row["seconds"] / (pool_size * passes) for row in rows]
    return {
        "seconds_per_request": {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "iqr": statistics.quantiles(values, n=4)[2] - statistics.quantiles(values, n=4)[0],
        },
        "ordered_seconds_per_request": values,
        "block_seconds": [row["seconds"] for row in rows],
        "median_block_seconds": statistics.median(row["seconds"] for row in rows),
        "pool_passes": passes,
        "requests_per_block": pool_size * passes,
        "sequence_sha256": rows[0]["sequence_sha256"],
    }
