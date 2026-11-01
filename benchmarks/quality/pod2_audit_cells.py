"""Recompute Pod 2 paired eligibility without trusting completion flags alone."""

import json
import math
import statistics

PIN = "97b741887dc0bd4730131aa60a7b5242fef8d5ef"
PACKAGES = {
    "fse": "fse",
    "model2vec": "model2vec",
    "rust": "model2vec-rs",
    "fastembed": "fastembed",
    "model2vec-classifier": "model2vec",
    "tfidf": "scikit-learn",
    "fasttext": "fasttext-numpy2",
}
IDENTITY = ("runtime_package", "runtime_package_version", "package_location", "runtime_fingerprint")


def stage(cell, identifier, probe):
    if identifier == probe:
        return "probe"
    parts = cell.split("/")
    if parts[0] == "encode":
        return parts[-4] if parts[-1] == "table" else "counterparts"
    return "pipelines" if parts[0] == "pipeline" else parts[0]


def expected_labels(cell):
    parts = cell.split("/")
    other = None if parts[0] == "encode" and parts[-1] == "table" else parts[-1]
    return {"fse-a": "fse", "fse-b": "fse", **({other: other} if other else {})}


def signature(row, fields):
    return json.dumps({key: row.get(key) for key in fields}, sort_keys=True)


def finite_seconds(row):
    value = row.get("p50_s")
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def round_status(rows, marker, expected):
    if not marker.get("complete"):
        return False, True
    if len(rows) != len(expected) or {r["label"] for r in rows} != set(expected):
        return False, False
    if not all(finite_seconds(r) for r in rows):
        return False, False
    controls = [r["p50_s"] for r in rows if r["label"] in ("fse-a", "fse-b")]
    ratio = max(controls) / min(controls)
    good = ratio <= 1.1 and len({r["input_digest"] for r in rows}) == 1
    matches = good == marker.get("valid")
    matches = matches and math.isclose(marker.get("aa_max_ratio", 0), ratio, rel_tol=1e-12)
    return good and marker.get("valid"), matches


def runtime_stable(rows, kind):
    attributed = all(
        r.get("runtime_package") == PACKAGES[kind] and all(r.get(k) for k in IDENTITY) for r in rows
    )
    return bool(rows) and attributed and len({signature(r, IDENTITY) for r in rows}) == 1


def arms_stable(operations, expected):
    for label, kind in expected.items():
        rows = [r for r in operations if r["label"] == label]
        if not runtime_stable(rows, kind):
            return False
        if any(r["kind"] != kind for r in rows) or len({r["output_digest"] for r in rows}) != 1:
            return False
    return True


def controls_stable(rows):
    fields = (
        *IDENTITY,
        "encoder_fingerprint",
        "artifact_provenance",
        "table_dtype",
        "output_dtype",
        "model",
        "revision",
        "input_digest",
        "texts",
    )
    controls = [r for r in rows if r["label"] in ("fse-a", "fse-b")]
    return (
        len({signature(r, fields) for r in controls}) == 1
        and len({r["output_digest"] for r in controls}) == 1
    )


def rate_summary(rows):
    rates = [r["texts"] / r["p50_s"] for r in rows]
    first = rows[0]
    result = {
        "rounds": [r["round"] for r in rows],
        "rates": rates,
        "median": statistics.median(rates),
        "min": min(rates),
        "max": max(rates),
        "runtime": {k: first.get(k) for k in IDENTITY},
    }
    for key in (
        "peak_rss_bytes",
        "file_bytes",
        "load_s",
        "first_encode_s",
        "first_predict_s",
        "p95_s",
        "time_until_durable_s",
    ):
        values = [r[key] for r in rows if r.get(key) is not None]
        result[key] = (
            {"min": min(values), "median": statistics.median(values), "max": max(values)}
            if values
            else None
        )
    return result


def paired_speedup(rows, label):
    pairs = {}
    for row in rows:
        pairs.setdefault(row["round"], {})[row["label"]] = row
    values = [pair[label]["p50_s"] / pair["fse-a"]["p50_s"] for pair in pairs.values()]
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "ratios": values,
    }


def summarize_arms(rows, expected):
    result = (
        {label: rate_summary([r for r in rows if r["label"] == label]) for label in expected}
        if rows
        else {}
    )
    for label, arm in result.items():
        if label not in ("fse-a", "fse-b"):
            arm["paired_speedup_over_reference"] = paired_speedup(rows, label)
    return result


def accepted_rounds(operations, markers, expected):
    accepted = []
    consistent = True
    for marker in markers:
        current = [r for r in operations if r["round"] == marker["round"]]
        good, matches = round_status(current, marker, expected)
        consistent = consistent and matches
        if good:
            accepted.append(marker["round"])
    return sorted(set(accepted)), consistent


def global_ratio(operations):
    controls = [r for r in operations if r["label"] in ("fse-a", "fse-b")]
    if not controls or not all(finite_seconds(r) for r in operations):
        return None
    return max(r["p50_s"] for r in controls) / min(r["p50_s"] for r in controls)


def source_and_budget(rows):
    fields = ("model", "revision", "machine", "affinity", "native_thread_limits")
    return (
        bool(rows)
        and all(r.get("commit") == PIN for r in rows)
        and len({signature(r, fields) for r in rows}) == 1
    )


def separate_backend_hashes(rows):
    hashes = {r.get("runtime_fingerprint") for r in rows if r["kind"] == "fse"}
    return all(r.get("runtime_fingerprint") not in hashes for r in rows if r["kind"] != "fse")


def gates_for(completions, markers, operations, accepted, consistent, expected):
    done = completions[0] if len(completions) == 1 else {}
    ratio = global_ratio(operations)
    return {
        "unique_completed_series": len(completions) == 1 and done.get("complete") is True,
        "three_complete_rounds": len(accepted) >= 3,
        "round_metadata_matches": consistent and len({r["round"] for r in markers}) == len(markers),
        "series_count_matches": done.get("valid_rounds") == len(accepted),
        "global_aa_within_10_percent": ratio is not None and ratio <= 1.1,
        "aa_record_matches": ratio is not None
        and math.isclose(done.get("aa_max_ratio", 0), ratio, rel_tol=1e-12),
        "own_backend_identity_and_stable_predictions": arms_stable(operations, expected),
        "identical_fse_controls": controls_stable(operations),
        "same_complete_inputs": len({(r["input_digest"], r["texts"]) for r in operations}) == 1,
        "same_source_and_budget": source_and_budget(operations),
        "counterpart_not_fse_identity": separate_backend_hashes(operations),
    }


def audit_metadata(operations, ratio):
    keys = (
        "model",
        "revision",
        "corpus",
        "texts",
        "input_digest",
        "table_dtype",
        "output_dtype",
        "cpu",
        "machine",
        "affinity",
        "requested_threads",
        "workers",
        "rss_scope",
    )
    noise = "incomplete" if ratio is None else "within 10 percent"
    if ratio and ratio > 1.1:
        noise = "noisy, re-measure in a quiet window"
    return {
        "measurement_date_utc": sorted(
            {r["time_utc"][:10] for r in operations if r.get("time_utc")}
        ),
        "first_record_context": {key: operations[0].get(key) for key in keys} if operations else {},
        "noise_label": noise,
    }


def audit(identifier, entries, probe):
    operations = [r for r in entries if r.get("event") == "operation"]
    markers = [r for r in entries if r.get("event") == "round_complete"]
    completions = [r for r in entries if r.get("event") == "series_complete"]
    cell = entries[0]["cell"]
    expected = expected_labels(cell)
    accepted, consistent = accepted_rounds(operations, markers, expected)
    selected = sorted([r for r in operations if r["round"] in accepted], key=lambda r: r["round"])
    ratio = global_ratio(operations)
    gates = gates_for(completions, markers, operations, accepted, consistent, expected)
    if stage(cell, identifier, probe) == "probe":
        gates["probe_duration"] = bool(completions) and completions[0].get("seconds", 0) >= 120
    return {
        "cell": cell,
        "series_id": identifier,
        "stage": stage(cell, identifier, probe),
        "eligible": all(gates.values()),
        "gates": gates,
        "accepted_rounds": accepted,
        "aa_upper": ratio,
        "symmetric_aa_band": [1 / ratio, ratio] if ratio else None,
        "arms": summarize_arms(selected, expected),
        **audit_metadata(operations, ratio),
        "rust_parallelism_bug_affected": False if "rust" in expected else None,
        "rust_bug_scope": "ea17433 changed only Pod3 large_write_paths; Pod2 encode and pipeline set both Rayon/tokenizer flags",
    }
