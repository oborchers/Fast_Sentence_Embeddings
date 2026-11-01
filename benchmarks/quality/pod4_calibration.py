"""Bounded per-model input sizing, retaining every observed memory rejection."""

import json
import math
import subprocess
import time

from compare_parallelism import child_environment
from pod4_inputs import write_input
from pod4_memory import MemoryAdmissionError
from pod4_storage import release_current
from pod_pairs import append
from pod_speed import encode_command, ensure_model

MINIMUM = 2.0
TARGET = 2.5
MAX_ROWS = 10_000_000


def bind_input(options, source, count):
    path, record = write_input(
        options.inputs, source, count, receipt_archive=options.output / "input-identities"
    )
    options.input_file, options.input_record = path, record
    return count


def calibration(options, repo, dtype, workers):
    ensure_model(repo, dtype, options)
    command = [*encode_command(options, repo, dtype, "wikitext", workers, "fse"), "--passes", "1"]
    done = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=max(0.01, options.deadline - time.monotonic()),
        env=child_environment(workers),
    )
    record = {
        "model": repo,
        "dtype": dtype,
        "workers": workers,
        "command": command,
        "returncode": done.returncode,
        "scope": "input sizing only, not a published speed cell",
    }
    if done.returncode:
        append(
            options.output / "calibration.jsonl",
            {**record, "stderr": done.stderr[-12000:], "stdout": done.stdout[-12000:]},
        )
        if done.returncode == 78:
            raise MemoryAdmissionError(json.loads(done.stdout.strip().splitlines()[-1])["budget"])
        raise RuntimeError("fse calibration child failed")
    row = json.loads(done.stdout.strip().splitlines()[-1])
    append(options.output / "calibration.jsonl", {**record, "measurement": row})
    return min(row["first_encode_s"], *row["warm_seconds"])


def midpoint(low, high):
    count = ((low + high) // 20_000) * 10_000
    if not low < count < high:
        raise MemoryError("no input size satisfies the unchanged duration and memory gates")
    return count


def selected(options, count, seconds, constrained):
    append(
        options.output / "calibration-decisions.jsonl",
        {
            "count": count,
            "minimum_actual_fse_call_seconds": seconds,
            "required_seconds": MINIMUM,
            "preferred_seconds": TARGET,
            "memory_constrained": constrained,
            "scope": "sizing only; every measured call is independently rechecked",
        },
    )
    return count


def attempt(options, specifications):
    try:
        return min(calibration(options, *spec) for spec in specifications)
    except MemoryAdmissionError:
        return None


def advance(count, low, high, seconds):
    if seconds is None:
        high = count if high is None else min(high, count)
        return midpoint(low, high), low, high, False
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("calibration returned a nonpositive duration")
    if seconds >= TARGET or (high is not None and seconds >= MINIMUM):
        return count, low, high, True
    low = count
    desired = min(
        MAX_ROWS, max(count + 10_000, math.ceil(count * TARGET / seconds * 1.1 / 10_000) * 10_000)
    )
    count = min(desired, midpoint(low, high)) if high is not None else desired
    if count <= low:
        raise MemoryError("maximum input cannot satisfy the two-second duration gate")
    return count, low, high, False


def calibrate(options, source, specifications):
    if not specifications:
        raise ValueError("calibration requires at least one specification")
    count, low, high = 250_000, 0, None
    for _ in range(12):
        if time.monotonic() >= options.deadline:
            raise TimeoutError("input calibration reached stage deadline")
        bind_input(options, source, count)
        seconds = attempt(options, specifications)
        next_count, low, high, accepted = advance(count, low, high, seconds)
        if accepted:
            return selected(options, count, seconds, high is not None)
        release_current(options)
        count = next_count
    raise MemoryError("bounded calibration cannot satisfy both duration and memory gates")
