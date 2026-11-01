"""Measured classifier rates, paired speedups and strict source/A/A eligibility."""

import json
import math
import statistics
from pathlib import Path

from pod_artifacts import emit


def spread(values):
    return (
        {"median": statistics.median(values), "min": min(values), "max": max(values)}
        if values
        else None
    )


def runtime(row):
    keys = ("runtime_package", "runtime_package_version", "package_location", "runtime_fingerprint")
    return {key: row.get(key) for key in keys}


def runtime_eligible(row):
    expected = "model2vec" if row["kind"] in ("mlp", "model2vec-linear") else "fse"
    return row.get("runtime_package") == expected and all(runtime(row).values())


def head_runtime_stable(rows):
    if rows[0]["kind"] != "model2vec-linear":
        return True
    heads = [row.get("linear_head_runtime") or {} for row in rows]
    valid = all(
        head.get("runtime_package") == "scikit-learn" and all(runtime(head).values())
        for head in heads
    )
    return valid and len({json.dumps(runtime(head), sort_keys=True) for head in heads}) == 1


def arm(rows):
    first = rows[0]
    stable = len({json.dumps(runtime(row), sort_keys=True) for row in rows}) == 1
    stable = stable and all(runtime_eligible(row) for row in rows) and head_runtime_stable(rows)
    return {
        "runtime": runtime(first),
        "runtime_stable": stable,
        "predictions_stable": len({row["output_digest"] for row in rows}) == 1,
        "head": first["head"],
        "normalization": first["normalization"],
        "linear_head_runtime": first.get("linear_head_runtime"),
        "linear_head_runtime_stable": head_runtime_stable(rows),
        "texts_per_second": spread([row["texts"] / row["p50_s"] for row in rows]),
        "seconds": spread([row["p50_s"] for row in rows]),
        "accuracy": [row["quality"] for row in rows],
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
        "folded_model_file_bytes": first["folded_model_file_bytes"],
        "encoder_plus_head_file_bytes": first["encoder_plus_head_file_bytes"],
    }


def paired_speedups(rows, accepted, labels):
    pairs = {
        number: {row["label"]: row for row in rows if row["round"] == number} for number in accepted
    }
    result = {}
    for label in labels:
        if label in ("fse-a", "fse-b"):
            continue
        result[label] = spread(
            [
                pair[label]["p50_s"] / pair["fse-a"]["p50_s"]
                for pair in pairs.values()
                if label in pair
            ]
        )
    return result


def complete_rounds(operations, rounds, expected):
    accepted = set()
    for number in {row["round"] for row in rounds if row.get("valid") and row.get("complete")}:
        current = [row for row in operations if row["round"] == number]
        if len(current) == len(expected) and {row["label"] for row in current} == expected:
            accepted.add(number)
    return accepted


def evidence(operations, rounds):
    kinds = {row["kind"] for row in operations}
    expected = (
        {"fse-a", "fse-b", "model2vec-mlp"}
        if "mlp" in kinds
        else {"fse-a", "fse-b", "encode-matmul", "model2vec-same-linear"}
    )
    accepted = complete_rounds(operations, rounds, expected)
    times = [row["p50_s"] for row in operations if row["label"] in ("fse-a", "fse-b")]
    positive = bool(times) and all(math.isfinite(value) and value > 0 for value in times)
    ratio = max(times) / min(times) if positive else None
    equal_inputs = len({(row["input_digest"], row["texts"]) for row in operations}) == 1
    return accepted, {
        "complete_rounds": len(accepted),
        "all_control_max_ratio": ratio,
        "symmetric_aa_band": [1 / ratio, ratio] if ratio is not None else None,
        "equal_inputs": equal_inputs,
        "eligible": len(accepted) >= 3 and ratio is not None and ratio <= 1.1 and equal_inputs,
    }


def summarize_cell(identifier, completed, operations, rounds):
    accepted, observed = evidence(operations, rounds)
    rows = [row for row in operations if row["round"] in accepted]
    if not rows:
        return {
            "series_id": identifier,
            "cell": completed["cell"],
            "eligible": False,
            "reason": "no valid complete rounds",
            "aa": completed,
            "verified_evidence": observed,
        }
    labels = sorted({row["label"] for row in rows})
    arms = {label: arm([row for row in rows if row["label"] == label]) for label in labels}
    first = rows[0]
    controls = [row for row in operations if row["label"] in ("fse-a", "fse-b")]
    stable_controls = (
        len({row["output_digest"] for row in controls}) == 1
        and len({json.dumps(runtime(row), sort_keys=True) for row in controls}) == 1
    )
    ratios = paired_speedups(rows, accepted, labels)
    eligible = (
        completed["complete"]
        and observed["eligible"]
        and completed.get("valid_rounds") == observed["complete_rounds"]
        and completed.get("aa_max_ratio") is not None
        and math.isclose(
            completed["aa_max_ratio"], observed["all_control_max_ratio"], rel_tol=1e-12
        )
        and stable_controls
        and all(a["runtime_stable"] and a["predictions_stable"] for a in arms.values())
    )
    return {
        "series_id": identifier,
        "cell": completed["cell"],
        "eligible": eligible,
        "source_commit": first["commit"],
        "dataset": first["dataset"],
        "suite": first["suite"],
        "dataset_revision": first["training_record"]["dataset_revision"],
        "model": first["model"],
        "revision": first["revision"],
        "table_dtype": "float32",
        "requested_workers": first["requested_workers"],
        "texts": first["texts"],
        "input_digest": first["input_digest"],
        "machine": first["machine"],
        "cpu": first["cpu"],
        "dates_utc": sorted({row["time_utc"][:10] for row in rows}),
        "primary_fse_summary": "fse-a; fse-b reported separately",
        "arms": arms,
        "paired_fse_folded_speedup_vs_counterpart": ratios,
        "aa": completed,
        "verified_evidence": observed,
        "control_predictions_stable": stable_controls,
        "note": "fine-tuned Model2Vec MLP is a different trained model; model2vec-same-linear uses the same sklearn coefficients",
    }


def summarize(rows):
    result = []
    for done in rows:
        if done.get("event") != "series_complete":
            continue
        identifier = done["series_id"]
        current = [row for row in rows if row.get("series_id") == identifier]
        operations = [row for row in current if row.get("event") == "operation"]
        rounds = [row for row in current if row.get("event") == "round_complete"]
        result.append(summarize_cell(identifier, done, operations, rounds))
    return {"format": "classifier-folding-summary-v1", "cells": result}


def publish_summary(folder: Path):
    source = folder / "speed-results.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    result = summarize(rows)
    (folder / "classification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    emit("classification-summary-" + str(len(result["cells"])), result)
    return result
