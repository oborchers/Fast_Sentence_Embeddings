"""Strict paired README evidence, preserving ordered primary rates and source contracts."""

import json
import math

from pod_folded_summary import complete_rounds, runtime, spread
from pod_large_summary import aa

PACKAGES = {
    "fse": "fse",
    "model2vec": "model2vec",
    "rust": "model2vec-rs",
    "fastembed": "fastembed",
    "st": "sentence-transformers",
}


def arm(rows, checked):
    first = rows[0]
    stable_code = len({json.dumps(runtime(row), sort_keys=True) for row in rows}) == 1
    attributed = all(
        row.get("runtime_package") == PACKAGES[row["kind"]] and all(runtime(row).values())
        for row in rows
    )
    reference = checked["records"][first["kind"]]
    correct_output = all(row["output_digest"] == reference["output_digest"] for row in rows)
    return {
        "backend": first["kind"],
        "runtime": runtime(first),
        "runtime_stable": stable_code and attributed and runtime(first) == runtime(reference),
        "output_matches_check": correct_output,
        "ordered_rounds": [row["round"] for row in rows],
        "ordered_texts_per_second": [row["texts"] / row["p50_s"] for row in rows],
        "texts_per_second": spread([row["texts"] / row["p50_s"] for row in rows]),
        "seconds": spread([row["p50_s"] for row in rows]),
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
        "file_bytes": first["file_bytes"],
        "tokenizer_parallelism": first.get("tokenizer_parallelism"),
        "tokenizer_thread_count": first.get("tokenizer_thread_count"),
        "parallel_batch_pool": first.get("parallel_batch_pool"),
        "torch_num_threads": first.get("torch_num_threads"),
    }


def common_context(rows, checked):
    first = rows[0]
    keys = (
        "model",
        "revision",
        "input_digest",
        "texts",
        "corpus",
        "machine",
        "commit",
        "requested_threads",
        "output_dtype",
        "table_dtype",
    )
    stable = all(
        len({json.dumps(row.get(key), sort_keys=True) for row in rows}) == 1 for key in keys
    )
    stable = (
        stable
        and first["input_digest"] == checked["input_digest"]
        and first["model"] == checked["model"]
        and first["revision"] == checked["revision"]
        and first["requested_threads"] == checked["workers"]
        and first["output_dtype"] == checked["output_dtype"]
        and first["table_dtype"] == checked["table_dtype"]
    )
    stable = stable and all(row["shape"] == checked["shape"] for row in rows)
    stable = stable and len({row["time_utc"][:10] for row in rows}) == 1
    stable = stable and all(
        row.get("encoder_fingerprint") == checked["records"]["fse"].get("encoder_fingerprint")
        for row in rows
        if row["kind"] == "fse"
    )
    return stable


def paired_ratios(rows, against):
    rounds = sorted({row["round"] for row in rows})
    values = []
    for number in rounds:
        pair = {row["label"]: row for row in rows if row["round"] == number}
        values.append(pair[against]["p50_s"] / pair["fse-a"]["p50_s"])
    return {
        **spread(values),
        "ordered_ratios": values,
        "convention": "median of paired reference-time/fse-a-time ratios",
    }


def eligible(done, checked, accepted, ratio, control_match, rows, arms):
    good = (
        done["complete"]
        and checked["passed"]
        and len(accepted) == 3
        and done.get("valid_rounds") == len(accepted)
        and ratio is not None
        and ratio <= 1.1
        and math.isclose(done.get("aa_max_ratio", 0), ratio, rel_tol=1e-12)
        and control_match
        and bool(rows)
        and common_context(rows, checked)
        and all(a["runtime_stable"] and a["output_matches_check"] for a in arms.values())
    )
    return good


def summarize(done, operations, rounds, labels, checked):
    accepted = complete_rounds(operations, rounds, set(labels))
    rows = sorted(
        [row for row in operations if row["round"] in accepted], key=lambda row: row["round"]
    )
    arms = (
        {label: arm([row for row in rows if row["label"] == label], checked) for label in labels}
        if rows
        else {}
    )
    ratio, control_match = aa(operations)
    good = eligible(done, checked, accepted, ratio, control_match, rows, arms)
    result = {
        "cell": done["cell"],
        "series_id": done["series_id"],
        "eligible": good,
        "aa": done,
        "verified_aa_ratio": ratio,
        "symmetric_aa_band": [1 / ratio, ratio] if ratio else None,
        "accepted_rounds": sorted(accepted),
        "arms": arms,
        "output_checked": checked,
        "primary_summary": "fse-a; fse-b separate",
        "publication_commit": None,
        "note": "Raw rows must be committed and independently audited before filling the README renderer input.",
    }
    result.update(measurement_fields(rows, labels))
    return result


def measurement_fields(rows, labels):
    result = {}
    if rows:
        first = rows[0]
        result["measurement"] = {
            key: first.get(key)
            for key in (
                "commit",
                "model",
                "revision",
                "input_digest",
                "corpus",
                "texts",
                "machine",
                "cpu",
                "python",
                "versions",
                "requested_threads",
                "output_dtype",
                "table_dtype",
            )
        }
        result["dates_utc"] = sorted({row["time_utc"][:10] for row in rows})
        result["paired_speedups"] = {
            label: paired_ratios(rows, label) for label in labels if label not in ("fse-a", "fse-b")
        }
    return result
