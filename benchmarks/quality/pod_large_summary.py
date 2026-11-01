"""Each write path stays distinct; only identical fse controls establish A/A noise."""

import json
import math

from pod_folded_summary import complete_rounds, runtime, spread


def arm(rows):
    first = rows[0]
    expected = "model2vec-rs" if first["kind"] == "rust" else "fse"
    attributed = all(
        row.get("runtime_package") == expected and all(runtime(row).values()) for row in rows
    )
    stable = len({json.dumps(runtime(row), sort_keys=True) for row in rows}) == 1
    return {
        "runtime": runtime(first),
        "runtime_stable": attributed and stable,
        "predictions_stable": len({row["output_digest"] for row in rows}) == 1,
        "texts_per_second": spread([row["texts_per_second"] for row in rows]),
        "tokens_per_second": spread([row["tokens_per_second"] for row in rows]),
        "seconds": spread([row["p50_s"] for row in rows]),
        "time_until_durable_s": spread(
            [row["time_until_durable_s"] for row in rows if row["time_until_durable_s"] is not None]
        ),
        "bytes_written": spread([row["bytes_written"] for row in rows]),
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
        "mode": first["mode"],
        "output_dtype": first["output_dtype"],
        "durability": first["durability"],
        "workers": first["requested_workers"],
    }


def aa(operations):
    controls = [row for row in operations if row["label"] in ("fse-a", "fse-b")]
    seconds = [row["p50_s"] for row in controls]
    positive = bool(seconds) and all(math.isfinite(value) and value > 0 for value in seconds)
    ratio = max(seconds) / min(seconds) if positive else None
    same_code = len({json.dumps(runtime(row), sort_keys=True) for row in controls}) == 1
    same_output = len({row["output_digest"] for row in controls}) == 1
    return ratio, same_code and same_output


def summarize(done, operations, rounds, expected, source):
    accepted = complete_rounds(operations, rounds, set(expected))
    selected = [row for row in operations if row["round"] in accepted]
    arms = (
        {label: arm([row for row in selected if row["label"] == label]) for label in expected}
        if selected
        else {}
    )
    ratio, controls_match = aa(operations)
    equal_inputs = len({row["input_digest"] for row in operations}) == 1
    eligible = (
        done["complete"]
        and len(accepted) >= 3
        and ratio is not None
        and ratio <= 1.1
        and math.isclose(done.get("aa_max_ratio", 0), ratio, rel_tol=1e-12)
        and done.get("valid_rounds") == len(accepted)
        and controls_match
        and equal_inputs
        and all(row["runtime_stable"] and row["predictions_stable"] for row in arms.values())
    )
    return {
        "cell": done["cell"],
        "eligible": eligible,
        "aa": done,
        "verified_aa_ratio": ratio,
        "symmetric_aa_band": [1 / ratio, ratio] if ratio else None,
        "valid_complete_pairs": len(accepted),
        "input_identity": source,
        "arms": arms,
        "primary_summary": "fse-a; fse-b separate; all paths have their own medians",
        "measurement": {
            key: operations[0].get(key)
            for key in (
                "commit",
                "cpu",
                "machine",
                "python",
                "versions",
                "time_utc",
                "model",
                "revision",
            )
        }
        if operations
        else {},
    }
