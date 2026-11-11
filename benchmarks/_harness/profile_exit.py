"""Recognize the pinned py-spy post-write child-wait failure without hiding it."""

from __future__ import annotations

import math
import re

from _harness.records import document

SOURCE = "https://github.com/benfred/py-spy/blob/v0.4.2/src/main.rs#L417-L424"
ERROR = "Error: No child process (os error 10)"


def echild_warning(spec, folder, status, meta):
    """Defer only this known post-write error to data and provenance verification."""
    if spec["tool"] != "py-spy" or status != 1:
        return None
    if meta["deadline"] or meta["escalated"] is not None or meta["signal"] is not None:
        return None
    if document(folder / "profiler.json")["version"] != "py-spy 0.4.2":
        return None
    log = (folder / "profile.log").read_text()
    suffix = re.search(
        r"py-spy> Stopped sampling because process exited\n"
        r"py-spy> Wrote speedscope file to '([^'\n]+)'\. Samples: (\d+) Errors: (\d+)\n"
        r"py-spy> Visit https://www\.speedscope\.app/ to view\n" + re.escape(ERROR) + r"\s*\Z",
        log,
    )
    if suffix is None or log.count("Error:") != 1:
        return None
    if suffix[1] != str(folder / "cpu.speedscope.json"):
        return None
    return {
        "type": "py_spy_post_write_echild",
        "exit_status": status,
        "message": ERROR,
        "source": SOURCE,
        "reported_samples": int(suffix[2]),
        "sampling_errors": int(suffix[3]),
        "reason": "child wait failed after writing; accepted only after data and provenance checks",
    }


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _frame(frame):
    if not isinstance(frame, dict) or not isinstance(frame.get("name"), str):
        raise ValueError("ECHILD recovery requires named frames")
    if frame.get("file") is not None and not isinstance(frame["file"], str):
        raise ValueError("ECHILD recovery found invalid frame file metadata")
    for key in ("line", "col"):
        value = frame.get(key)
        if value is not None and (type(value) is not int or not 0 <= value < 2**32):
            raise ValueError("ECHILD recovery found invalid frame position metadata")


def _metadata(data, profiles):
    if data.get("name") is not None and not isinstance(data["name"], str):
        raise ValueError("ECHILD recovery found invalid file name metadata")
    index = data.get("activeProfileIndex")
    if index is not None and (not _number(index) or index % 1 != 0 or index >= len(profiles)):
        raise ValueError("ECHILD recovery found invalid active profile metadata")


def _sampled(profile, frames):
    if not isinstance(profile["name"], str):
        raise ValueError("ECHILD recovery requires named profiles")
    if profile["type"] != "sampled" or profile["unit"] != "seconds":
        raise ValueError("ECHILD recovery requires sampled seconds profiles")
    samples, weights = profile["samples"], profile["weights"]
    if (
        not isinstance(samples, list)
        or not isinstance(weights, list)
        or len(samples) != len(weights)
    ):
        raise ValueError("ECHILD recovery requires matching samples and weights")
    if not all(_number(w) and w > 0 for w in weights):
        raise ValueError("ECHILD recovery requires positive finite weights")
    for stack in samples:
        if not isinstance(stack, list) or not stack:
            raise ValueError("ECHILD recovery requires nonempty sample stacks")
        if any(type(i) is not int or not 0 <= i < len(frames) for i in stack):
            raise ValueError("ECHILD recovery found an invalid frame reference")
    start, end = profile["startValue"], profile["endValue"]
    if not _number(start) or not _number(end) or end < start:
        raise ValueError("ECHILD recovery found invalid profile bounds")
    if not math.isclose(sum(weights), end - start, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("ECHILD recovery found inconsistent sample duration")
    return len(samples)


def verify_echild_data(folder, receipt, warning):
    """Require complete sampled data and explicit evidence for the addressed child."""
    if receipt.get("error") or receipt.get("diagnostic_profile") is not True:
        raise ValueError("ECHILD recovery requires a successful diagnostic receipt")
    if receipt.get("valid_for_comparison") is not False:
        raise ValueError("ECHILD recovery requires a non-comparison receipt")
    data = document(folder / "cpu.speedscope.json")
    if (
        data["$schema"] != "https://www.speedscope.app/file-format-schema.json"
        or data["exporter"] != "py-spy@0.4.2"
    ):
        raise ValueError("ECHILD recovery requires the pinned Speedscope exporter")
    frames, profiles = data["shared"]["frames"], data["profiles"]
    if not isinstance(frames, list) or not frames or not isinstance(profiles, list) or not profiles:
        raise ValueError("ECHILD recovery requires frames and profiles")
    _metadata(data, profiles)
    for frame in frames:
        _frame(frame)
    count = sum(_sampled(profile, frames) for profile in profiles)
    if count <= 0 or count != warning["reported_samples"]:
        raise ValueError("ECHILD recovery sample count differs from the write confirmation")
    pid = receipt["diagnostic_pid"]
    child = re.compile(rf"(?:Process {pid} Thread |Thread {pid} )")
    if not any(child.match(p["name"]) and p["samples"] for p in profiles):
        raise ValueError("ECHILD recovery has no samples explicitly attributed to the child")
