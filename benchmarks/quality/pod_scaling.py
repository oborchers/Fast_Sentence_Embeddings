"""Final Pod 3 stage: the headline encoder pipeline at every requested worker count."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

from artifact_manifest import header
from compare_models import artifact_path
from pod_artifacts import emit, export_jsonl
from pod_folded_summary import complete_rounds, runtime, spread
from pod_pairs import append, series
from pod_speed import POTION, download_corpus, encode_command, ensure_model

WORKERS = (1, 2, 4, 6, 8, 10, 16, 32, 3, 5, 7, 9)
RESERVE_SECONDS = 300


def eligibility(done, operations, accepted):
    times = [row["p50_s"] for row in operations]
    positive = bool(times) and all(math.isfinite(value) and value > 0 for value in times)
    ratio = max(times) / min(times) if positive else None
    stable_runtime = len({json.dumps(runtime(row), sort_keys=True) for row in operations}) == 1
    attributed = all(
        row.get("runtime_package") == "fse" and all(runtime(row).values()) for row in operations
    )
    equal_inputs = len({row["input_digest"] for row in operations}) == 1
    equal_outputs = len({row["output_digest"] for row in operations}) == 1
    good = (
        done["complete"]
        and len(accepted) >= 3
        and ratio is not None
        and ratio <= 1.1
        and done.get("valid_rounds") == len(accepted)
        and math.isclose(done.get("aa_max_ratio", 0), ratio, rel_tol=1e-12)
        and stable_runtime
        and attributed
        and equal_inputs
        and equal_outputs
    )
    return good, ratio


def cell_summary(done, operations, rounds, stamp):
    accepted = complete_rounds(operations, rounds, {"fse-a", "fse-b"})
    rows = [row for row in operations if row["round"] in accepted]
    good, ratio = eligibility(done, operations, accepted)
    first = operations[0] if operations else {}
    return {
        "cell": done["cell"],
        "eligible": good,
        "aa": done,
        "verified_aa_ratio": ratio,
        "symmetric_aa_band": [1 / ratio, ratio] if ratio else None,
        "valid_complete_pairs": len(accepted),
        "configuration": {
            key: first.get(key)
            for key in (
                "model",
                "revision",
                "table_dtype",
                "output_dtype",
                "corpus",
                "texts",
                "input_digest",
                "output_digest",
                "encoder_fingerprint",
                "artifact_provenance",
                "fse_workers",
                "machine",
                "cpu",
                "python",
                "packages",
                "versions",
                "commit",
                "timed_passes",
                "first_encode_texts",
                "load_source",
                "file_bytes",
            )
        },
        "runtime": runtime(first),
        "primary_summary": "fse-a, fse-b reported separately",
        "arms": {
            label: spread([row["texts"] / row["p50_s"] for row in rows if row["label"] == label])
            for label in ("fse-a", "fse-b")
        },
        "peak_rss_bytes": max((row["peak_rss_bytes"] for row in rows), default=None),
        "measurement_interval_utc": stamp,
        "dates_utc": sorted({value[:10] for value in stamp.values()}),
    }


def summarize(folder):
    rows = [json.loads(line) for line in (folder / "speed-results.jsonl").read_text().splitlines()]
    stamps = [
        json.loads(line) for line in (folder / "stage-progress.jsonl").read_text().splitlines()
    ]
    finished = [row for row in rows if row.get("event") == "series_complete"]
    cells = [summarize_series(done, rows, stamps) for done in finished]
    return {"format": "worker-scaling-v1", "requested_workers": sorted(WORKERS), "cells": cells}


def summarize_series(done, rows, stamps):
    current = [row for row in rows if row.get("series_id") == done["series_id"]]
    stamp = next(row for row in stamps if row["series_id"] == done["series_id"])
    interval = {key: stamp[key] for key in ("stage_start_utc", "stage_end_utc")}
    return cell_summary(
        done,
        [row for row in current if row.get("event") == "operation"],
        [row for row in current if row.get("event") == "round_complete"],
        interval,
    )


def run(args):
    import time

    folder = args.output / "worker-scaling"
    folder.mkdir()
    plan = {
        "model": POTION,
        "table_dtype": "float32",
        "corpus": "wikitext",
        "workers": WORKERS,
        "scope": "encode only, same bench_compare.py child as the headline",
        "rust": "not included: building Rust would consume the reserved final-stage budget",
        "reserve_seconds": RESERVE_SECONDS,
    }
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    emit("worker-scaling-plan", plan)
    print("stage_start worker-scaling", flush=True)
    options = SimpleNamespace(
        models=folder / "models", cache=args.cache, output=folder, deadline=args.deadline
    )
    download_corpus("wikitext")
    ensure_model(POTION, "float32", options)
    plan["encoder_header"] = header(artifact_path(options.models, POTION, "float32"))
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for workers in WORKERS:
        if time.monotonic() >= args.deadline:
            raise TimeoutError("worker scaling reached the whole-pod deadline")
        commands = {
            label: encode_command(options, POTION, "float32", "wikitext", workers, "fse")
            for label in ("fse-a", "fse-b")
        }
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result = series(commands, workers, folder, args.deadline, f"worker-scaling/{workers}")
        result["stage_start_utc"] = started
        result["stage_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        append(folder / "stage-progress.jsonl", {"workers": workers, **result})
        summary = {**summarize(folder), "encoder_header": plan["encoder_header"]}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        emit("worker-scaling-summary-" + str(workers), summary)
    print("stage_end worker-scaling", flush=True)
    export_jsonl("worker-scaling-raw", folder / "speed-results.jsonl")
