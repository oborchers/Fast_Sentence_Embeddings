"""Summarize only completed series and complete valid paired rounds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


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
    return labels


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(path: Path, hourly_price: float) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pairs = completed_rounds(rows)
    if not pairs:
        return {"raw": str(path), "status": "incomplete", "valid_pairs": 0}
    other = next(label for label in pairs[0] if label not in ("fse-a", "fse-b"))
    noise = [pair["fse-b"]["p50_s"] / pair["fse-a"]["p50_s"] for pair in pairs]
    envelope = max(max(noise), max(1 / value for value in noise))
    backends = {}
    for label in pairs[0]:
        records = [pair[label] for pair in pairs]
        rates = [row["texts"] / row["p50_s"] for row in records]
        backends[label] = {
            "model": records[0]["model"],
            "revision": records[0]["revision"],
            "table_dtype": records[0]["table_dtype"],
            "output_dtype": records[0]["output_dtype"],
            "file_bytes": records[0]["file_bytes"],
            "texts_per_second": spread(rates),
            "peak_rss_bytes": max(row["peak_rss_bytes"] for row in records),
            "model_setup_seconds": spread([row["load_s"] for row in records]),
            "reference_tariff_eur_per_million": spread(
                [1e6 / rate / 3600 * hourly_price for rate in rates]
            ),
        }
    return {
        "raw": str(path),
        "status": "complete",
        "valid_pairs": len(pairs),
        "corpus": pairs[0]["fse-a"]["corpus"],
        "workers": pairs[0]["fse-a"]["threads"],
        "input_digest": pairs[0]["fse-a"]["input_digest"],
        "backends": backends,
        "paired_speed_ratio_fse_over_other": spread(
            [pair[other]["p50_s"] / pair["fse-a"]["p50_s"] for pair in pairs]
        ),
        "symmetric_aa_band": [1 / envelope, envelope],
        "load_min": min(
            min(row["load_before"], row["load_after"]) for pair in pairs for row in pair.values()
        ),
        "load_max": max(
            max(row["load_before"], row["load_after"]) for pair in pairs for row in pair.values()
        ),
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
