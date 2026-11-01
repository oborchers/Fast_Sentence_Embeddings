"""Large end-to-end cells, before the final worker-scaling stage."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from pod_artifacts import emit, export_jsonl
from pod_large_inputs import MATRIX_ROWS, prepare_one
from pod_large_summary import summarize
from pod_pairs import append, series
from pod_speed import POTION, ensure_model

RESERVE_SECONDS = 4200


def commands(base, workers):
    control = [
        *base,
        "--mode",
        "shard",
        "--storage-dtype",
        "float32",
        "--durability",
        "none",
    ]
    result = {label: control for label in ("fse-a", "fse-b")}
    for mode in ("encode", "memory"):
        result[mode] = [*base, "--mode", mode, "--durability", "none"]
    if workers > 1:
        result.update(shard_commands(base))
    for durability in ["none"] if workers == 1 else ["none", "full"]:
        result["rust-npy-" + durability] = [
            *base,
            "--kind",
            "rust",
            "--mode",
            "npy",
            "--durability",
            durability,
        ]
    return result


def shard_commands(base):
    result = {}
    for dtype in ("float32", "float16", "int8"):
        for durability in ("none", "full"):
            if dtype == "float32" and durability == "none":
                continue
            result[f"shard-{dtype}-{durability}"] = [
                *base,
                "--mode",
                "shard",
                "--storage-dtype",
                dtype,
                "--durability",
                durability,
            ]
    return result


def timed_cell(args, folder, inputs, models, source, workers):
    if time.monotonic() >= args.deadline:
        raise TimeoutError("large series reached its reserved deadline")
    corpus = Path(source["file"]).stem
    base = [
        sys.executable,
        str(Path(__file__).with_name("pod_large_child.py")),
        "--inputs",
        str(inputs),
        "--models",
        str(models),
        "--corpus",
        corpus,
        "--workers",
        str(workers),
    ]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    operation_commands = commands(base, workers)
    append(
        folder / "matrix-plan.jsonl",
        {"corpus": corpus, "workers": workers, "commands": operation_commands},
    )
    result = series(
        operation_commands,
        workers,
        folder,
        args.deadline,
        f"large/{corpus}/{workers}",
    )
    interval = {
        "stage_start_utc": started,
        "stage_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    append(folder / "stage-progress.jsonl", {**result, **interval})
    all_rows = [
        json.loads(line) for line in (folder / "speed-results.jsonl").read_text().splitlines()
    ]
    rows = [row for row in all_rows if row.get("series_id") == result["series_id"]]
    summary = summarize(
        result,
        [row for row in rows if row.get("event") == "operation"],
        [row for row in rows if row.get("event") == "round_complete"],
        operation_commands,
        source,
    )
    summary["measurement_interval_utc"] = interval
    return summary


def run(args):
    folder = args.output / "large-end-to-end"
    folder.mkdir()
    inputs = folder / "pipeline-inputs"
    options = SimpleNamespace(models=folder / "models", cache=args.cache)
    ensure_model(POTION, "float32", options)
    source = prepare_one(inputs, MATRIX_ROWS)
    (folder / "input-identities.json").write_text(json.dumps([source], indent=2) + "\n")
    emit("large-input-identity-10m", source)
    summaries = []
    for workers in (10, 32, 1):
        summary = timed_cell(args, folder, inputs, options.models, source, workers)
        summaries.append(summary)
        (folder / "summary.json").write_text(
            json.dumps({"format": "large-end-to-end-v1", "cells": summaries}, indent=2) + "\n"
        )
        emit(f"large-summary-10m-{workers}", summaries)
    export_jsonl("large-e2e-raw", folder / "speed-results.jsonl")
