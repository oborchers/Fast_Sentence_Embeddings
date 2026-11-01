"""Pod-only rotating pairs. Host load is evidence, fse A/A is the validity gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from compare_parallelism import child_environment
from pod_artifacts import environment
from pod_duration import qualifies
from pod_timeouts import abort


def append(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def aa_band(rows):
    controls = [row for row in rows if row["label"] in ("fse-a", "fse-b")]
    if len(controls) != 2 or any(row["p50_s"] <= 0 for row in controls):
        raise ValueError("round needs two positive fse control measurements")
    times = [row["p50_s"] for row in controls]
    ratio = max(times) / min(times)
    return {
        "aa_max_ratio": ratio,
        "symmetric_aa_band": [1 / ratio, ratio],
        "aa_spread_percent": (ratio - 1) * 100,
        "control_seconds": times,
        "noise_label": "within 10% A/A band"
        if ratio <= 1.1
        else "noisy, re-measure in a quiet window",
    }


def noise_label(ratio):
    if ratio is None:
        return "unavailable, incomplete"
    if ratio > 1.1:
        return "noisy, re-measure in a quiet window"
    return "within 10% A/A band"


def operation(command, label, workers, deadline, output, context, hardware):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("pod deadline reached")
    before = os.getloadavg()
    started = time.monotonic()
    done = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=remaining,
        env=child_environment(workers),
    )
    append(
        output / "invocations.jsonl",
        {
            **context,
            **hardware,
            "command": command,
            "status": "completed" if done.returncode == 0 else "failed",
            "returncode": done.returncode,
            "seconds": time.monotonic() - started,
            "stderr": done.stderr[-12000:],
        },
    )
    if done.returncode:
        raise RuntimeError(f"child failed ({done.returncode}): {done.stderr[-3000:]}")
    row = json.loads(done.stdout.strip().splitlines()[-1])
    return {
        **row,
        **hardware,
        **context,
        "label": label,
        "event": "operation",
        "load_before": before,
        "load_after": os.getloadavg(),
        "host_load_policy": "recorded, never rejects a shared pod run",
    }


def round_observation(commands, ordered, workers, deadline, output, context, hardware):
    rows = []
    for label in ordered:
        row = operation(commands[label], label, workers, deadline, output, context, hardware)
        append(output / "speed-results.jsonl", row)
        rows.append(row)
    band = aa_band(rows)
    equal_inputs = len({row["input_digest"] for row in rows}) == 1
    long_enough = all(
        qualifies(row)
        for row in rows
        if row["label"] in ("fse-a", "fse-b") or row.get("runtime_package") == "fse"
    )
    good = band["aa_max_ratio"] <= 1.1 and equal_inputs and long_enough
    append(
        output / "speed-results.jsonl",
        {
            **context,
            **hardware,
            **band,
            "event": "round_complete",
            "valid": good,
            "complete": True,
            "equal_inputs": equal_inputs,
            "minimum_call_duration_passed": long_enough,
        },
    )
    return good, band


def series(commands, workers, output, deadline, cell, *, minimum_seconds=0, abort_on_timeout=False):
    if sys.platform != "linux":
        raise RuntimeError("pod timing policy is restricted to Linux")
    hardware = environment()
    identifier = uuid.uuid4().hex
    start = time.monotonic()
    valid, ratios, controls = 0, [], []
    failed = False
    labels = list(commands)
    for attempt in range(1000 if minimum_seconds else 9):
        if time.monotonic() >= deadline:
            break
        ordered = labels[attempt % len(labels) :] + labels[: attempt % len(labels)]
        context = {"series_id": identifier, "cell": cell, "round": attempt}
        try:
            good, band = round_observation(
                commands, ordered, workers, deadline, output, context, hardware
            )
            ratios.append(band["aa_max_ratio"])
            controls.extend(band["control_seconds"])
            valid += good
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
            failed = True
            append(
                output / "speed-results.jsonl",
                {
                    **context,
                    **hardware,
                    "event": "round_complete",
                    "valid": False,
                    "complete": False,
                    "reason": repr(error),
                },
            )
            abort(error, abort_on_timeout)
            break
        if valid >= 3 and time.monotonic() - start >= minimum_seconds:
            break
    return finish_series(
        output, hardware, identifier, cell, valid, ratios, controls, failed, start, minimum_seconds
    )


def finish_series(
    output, hardware, identifier, cell, valid, ratios, controls, failed, start, minimum_seconds
):
    # Probe rejection uses every observed A/A round, including noisy rounds.
    elapsed = time.monotonic() - start
    control_ratio = max(controls) / min(controls) if controls else None
    result = {
        "event": "series_complete",
        "series_id": identifier,
        "cell": cell,
        "valid_rounds": valid,
        "required_rounds": 3,
        "complete": not failed
        and valid >= 3
        and bool(ratios)
        and max(ratios) <= 1.1
        and control_ratio <= 1.1
        and elapsed >= minimum_seconds,
        "aa_max_ratio": control_ratio,
        "aa_within_round_max_ratio": max(ratios) if ratios else None,
        "symmetric_aa_band": [1 / control_ratio, control_ratio] if controls else None,
        "noise_label": noise_label(control_ratio),
        "incomplete_round": failed,
        "aa_control_max_ratio": control_ratio,
        "aa_control_seconds_range": [min(controls), max(controls)] if controls else None,
        "minimum_observation_seconds": minimum_seconds,
        "seconds": elapsed,
        **hardware,
    }
    append(output / "speed-results.jsonl", result)
    return result
