"""Summarize only completed series and complete valid paired rounds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from compare_compatibility import identity_status, require_compatible, signature


def completed_rounds(rows: list[dict]) -> list[dict[str, dict]]:
    finished = [
        row
        for row in rows
        if row["event"] == "series_complete" and row["valid_rounds"] >= row["required_rounds"] >= 3
    ]
    if not finished:
        return []
    series = finished[-1]["series_id"]
    valid = {
        row["round"]
        for row in rows
        if row["series_id"] == series
        and row["event"] == "round_complete"
        and row["complete"]
        and row["valid"]
    }
    pairs = []
    for number in sorted(valid):
        operations = [
            row
            for row in rows
            if row["series_id"] == series and row["event"] == "operation" and row["round"] == number
        ]
        pairs.append(validate_pair(operations))
    if len(pairs) < finished[-1]["required_rounds"]:
        raise ValueError("series completion count is not backed by valid rounds")
    require_compatible(pairs)
    return pairs


def validate_pair(operations: list[dict]) -> dict[str, dict]:
    labels = {row["label"]: row for row in operations}
    if len(operations) != 3 or len(labels) != 3 or not {"fse-a", "fse-b"} <= labels.keys():
        raise ValueError("completed round does not contain three distinct expected operations")
    if not all(
        row["valid"] and row["load_before"] < 5 and row["load_after"] < 5 for row in operations
    ):
        raise ValueError("completed round has an invalid operation")
    if len({row["input_digest"] for row in operations}) != 1:
        raise ValueError("completed round has different inputs")
    require_compatible([labels])
    return labels


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def backend_summary(records: list[dict], hourly_price: float) -> dict:
    rates = [row["texts"] / row["p50_s"] for row in records]
    return {
        "model": records[0]["model"],
        "revision": records[0]["revision"],
        "table_dtype": records[0]["table_dtype"],
        "output_dtype": records[0]["output_dtype"],
        "file_bytes": records[0]["file_bytes"],
        "fse_workers": records[0].get("fse_workers"),
        "tokenizer_parallelism": records[0].get("tokenizer_parallelism"),
        "tokenizer_thread_count": None,
        "parallel_batch_pool": records[0].get("parallel_batch_pool"),
        "texts_per_second": spread(rates),
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in records),
        "model_setup_seconds": spread([row["load_s"] for row in records]),
        "reference_tariff_eur_per_million": spread(
            [1e6 / rate / 3600 * hourly_price for rate in rates]
        ),
    }


def loads(pairs: list[dict]) -> tuple[float, float]:
    endpoints = []
    for pair in pairs:
        for row in pair.values():
            endpoints.extend((row["load_before"], row["load_after"]))
    return min(endpoints), max(endpoints)


def aa_summary(pairs: list[dict]) -> dict:
    ratios = [pair["fse-b"]["p50_s"] / pair["fse-a"]["p50_s"] for pair in pairs]
    if not ratios:
        return {
            "aa_ratio_b_over_a": None,
            "symmetric_aa_band": None,
            "noise_status": "unavailable, incomplete",
        }
    envelope = max(max(ratios), max(1 / value for value in ratios))
    return {
        "aa_ratio_b_over_a": spread(ratios),
        "symmetric_aa_band": [1 / envelope, envelope],
        "noise_status": (
            "noisy, re-measure in a quiet window" if envelope > 1.1 else "within 10% A/A band"
        ),
    }


def diagnostic_aa_groups(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple, dict] = {}
    for row in rows:
        if (
            row["event"] == "operation"
            and row.get("label") in ("fse-a", "fse-b")
            and row["valid"]
            and row["load_before"] < 5
            and row["load_after"] < 5
        ):
            grouped.setdefault((row["series_id"], row["round"]), {})[row["label"]] = row
    pairs = [
        pair
        for pair in grouped.values()
        if len(pair) == 2 and pair["fse-a"]["input_digest"] == pair["fse-b"]["input_digest"]
    ]
    for pair in pairs:
        require_compatible([pair])
    configurations: dict[str, list[dict]] = {}
    for pair in pairs:
        configurations.setdefault(signature(pair["fse-a"]), []).append(pair)
    return list(configurations.values())


def incomplete_aa(rows: list[dict]) -> list[dict]:
    groups = diagnostic_aa_groups(rows)
    return groups[-1] if groups else []


def diagnostic_aa(rows: list[dict]) -> list[dict]:
    return [
        {
            "configuration": json.loads(signature(group[0]["fse-a"])),
            "valid_pairs": len(group),
            **aa_summary(group),
        }
        for group in diagnostic_aa_groups(rows)
    ]


def summarize(path: Path, hourly_price: float) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pairs = completed_rounds(rows)
    if not pairs:
        return {
            "raw": str(path),
            "status": "incomplete",
            "valid_pairs": 0,
            "diagnostic_aa_configurations": diagnostic_aa(rows),
            **aa_summary(incomplete_aa(rows)),
            "aa_scope": "diagnostic valid A/A operations, incomplete speed series",
        }
    other = next(label for label in pairs[0] if label not in ("fse-a", "fse-b"))
    backends = {
        label: backend_summary([pair[label] for pair in pairs], hourly_price) for label in pairs[0]
    }
    load_min, load_max = loads(pairs)
    return {
        "raw": str(path),
        "status": "complete",
        "identity_status": identity_status(pairs),
        "diagnostic_aa_configurations": diagnostic_aa(rows),
        "valid_pairs": len(pairs),
        "corpus": pairs[0]["fse-a"]["corpus"],
        "workers": pairs[0]["fse-a"]["threads"],
        "input_digest": pairs[0]["fse-a"]["input_digest"],
        "backends": backends,
        "paired_speed_ratio_fse_over_other": spread(
            [pair[other]["p50_s"] / pair["fse-a"]["p50_s"] for pair in pairs]
        ),
        **aa_summary(pairs),
        "aa_scope": "complete valid paired series",
        "load_min": load_min,
        "load_max": load_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument("--price", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    price = json.loads(args.price.read_text())
    result = {
        "reference_tariff": price,
        "cells": [summarize(path, price["hourly_price"]) for path in args.journals],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
