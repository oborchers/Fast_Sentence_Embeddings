"""Report complete search lifecycles, recall and A/A controls without hiding noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_compare_report import aa_summary, completed_rounds, incomplete_aa, spread

PHASES = (
    "load_s",
    "encode_s",
    "build_s",
    "save_s",
    "index_build_and_save_s",
    "open_s",
    "first_search_s",
    "p50_s",
    "p95_s",
    "batch_p50_s",
    "batch_p95_s",
)


def backend(records):
    first = records[0]
    return {
        "kind": first["kind"],
        "index_settings": first["index_settings"],
        "encoder_start": first["encoder_start"],
        "table_load_build_scope": first["table_load_build_scope"],
        "phase_seconds": {key: spread([row[key] for row in records]) for key in PHASES},
        "table_load_build_seconds": spread([row["table_load_build_s"] for row in records])
        if first["table_load_build_s"] is not None
        else None,
        "recall_at_k": spread([row["recall_at_k"] for row in records]),
        "boundary_equivalent_recall_at_k": spread(
            [row["boundary_equivalent_recall_at_k"] for row in records]
        ),
        "queries_with_boundary_ties": [row["queries_with_boundary_ties"] for row in records],
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in records),
        "rss_scope": first["rss_scope"],
        "file_bytes": [row["file_bytes"] for row in records],
        "durability": first["durability"],
    }


def require_compatible(records):
    common = (
        "commit",
        "runtime_fingerprint",
        "versions",
        "machine",
        "python",
        "threads",
        "query_batch",
        "k",
        "encoder_start",
        "model",
        "revision",
        "encoder_fingerprint",
        "shape",
        "table_dtype",
        "input_digest",
        "vector_digest",
        "durability",
    )
    baseline = records[0]
    if any(
        any(key not in row or row[key] != baseline.get(key) for key in common) for row in records
    ):
        raise ValueError("search series contains incompatible source/runtime/settings identities")
    for label in {row["label"] for row in records}:
        same = [row for row in records if row["label"] == label]
        if any(
            row["kind"] != same[0]["kind"] or row["index_settings"] != same[0]["index_settings"]
            for row in same
        ):
            raise ValueError("search series changed backend settings")


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pairs = completed_rounds(rows)
    if not pairs:
        return {"raw": str(path), "status": "incomplete", **aa_summary(incomplete_aa(rows))}
    records = [row for pair in pairs for row in pair.values()]
    require_compatible(records)
    if len({row["vector_digest"] for row in records}) != 1:
        raise ValueError("complete search series changed its vector matrix")
    first = pairs[0]["fse-a"]
    other = next(label for label in pairs[0] if label not in ("fse-a", "fse-b"))
    return {
        "raw": str(path),
        "status": "complete",
        "valid_pairs": len(pairs),
        "primary_fse": "fse-a",
        "shape": first["shape"],
        "query_batch": first["query_batch"],
        "k": first["k"],
        "workers": first["threads"],
        "source_model": first["model"],
        "source_revision": first["revision"],
        "vector_digest": first["vector_digest"],
        "input_digest": first["input_digest"],
        "oracle": first["oracle"],
        "backends": {label: backend([pair[label] for pair in pairs]) for label in pairs[0]},
        "paired_warm_time_ratio_other_over_fse": spread(
            [pair[other]["p50_s"] / pair["fse-a"]["p50_s"] for pair in pairs]
        ),
        "load_min": min(min(row["load_before"], row["load_after"]) for row in records),
        "load_max": max(max(row["load_before"], row["load_after"]) for row in records),
        **aa_summary(pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            {"cells": [summarize(path) for path in args.journals]}, indent=2, allow_nan=False
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
