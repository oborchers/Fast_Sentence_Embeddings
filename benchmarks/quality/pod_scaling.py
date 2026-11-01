"""Final Pod 3 stage: the headline encoder pipeline at every requested worker count."""

from __future__ import annotations

import json
import math

from pod_artifacts import emit
from pod_folded_summary import complete_rounds, runtime, spread
from pod_speed import POTION

WORKERS = (1, 2, 4, 8, 16, 32)


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
    """Export both curves already measured with the stricter README output checks."""
    folder = args.output / "worker-scaling"
    folder.mkdir()
    source = args.output / "readme-panels" / "summary.json"
    report = json.loads(source.read_text())
    models = (POTION, "sentence-transformers/static-retrieval-mrl-en-v1")
    wanted = {
        f"encode/{repo}/float32/wikitext/{workers}/table" for repo in models for workers in WORKERS
    }
    cells = [cell for cell in report["cells"] if cell["cell"] in wanted]
    result = {
        "format": "worker-scaling-v2",
        "models": models,
        "requested_workers": WORKERS,
        "cells": cells,
        "missing": sorted(wanted - {cell["cell"] for cell in cells}),
        "source": str(source),
        "note": "Reuses the exact paired README series, without a second timing run.",
    }
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    emit("worker-scaling-summary", result)
