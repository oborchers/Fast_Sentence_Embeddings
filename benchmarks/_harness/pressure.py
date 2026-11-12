"""Optional CPU pressure observations and report-only noise labels, never admission gates."""

from __future__ import annotations

import math
import re
import statistics
import sys
import time
from pathlib import Path, PurePosixPath

UNAVAILABLE = "unavailable"


def _text(path):
    try:
        return path.read_text()
    except (OSError, UnicodeError):
        return None


def _unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup(proc):
    """Resolve this process's unified cgroup through its visible cgroup2 mount."""
    membership = _text(proc / "self/cgroup") or ""
    mounts = _text(proc / "self/mountinfo") or ""
    groups = [line[3:] for line in membership.splitlines() if line.startswith("0::")]
    if len(groups) != 1 or ".." in PurePosixPath(groups[0]).parts:
        return None
    group = PurePosixPath(groups[0])
    candidates = [
        (len(root.parts), mount / str(group.relative_to(root)))
        for root, mount in _mounts(mounts)
        if group.is_relative_to(root)
    ]
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def _mounts(text):
    for line in text.splitlines():
        fields = line.split()
        if "-" not in fields or len(fields) < 7:
            continue
        separator = fields.index("-")
        if separator + 1 < len(fields) and fields[separator + 1] == "cgroup2":
            yield PurePosixPath(_unescape(fields[3])), Path(_unescape(fields[4]))


def _pressure(text):
    if text is None:
        return UNAVAILABLE
    result = {}
    try:
        for line in text.splitlines():
            kind, *fields = line.split()
            values = dict(field.split("=", 1) for field in fields)
            row = {name: float(values[name]) for name in ("avg10", "avg60", "avg300")}
            row["total"] = int(values["total"])
            if any(not math.isfinite(v) or v < 0 for v in row.values()):
                return UNAVAILABLE
            result[kind] = row
    except (ValueError, KeyError):
        return UNAVAILABLE
    return result if "some" in result else UNAVAILABLE


def _stat(text):
    if text is None:
        return UNAVAILABLE
    try:
        result = {name: int(value) for name, value in (line.split() for line in text.splitlines())}
    except ValueError:
        return UNAVAILABLE
    return result if result and all(v >= 0 for v in result.values()) else UNAVAILABLE


def snapshot(proc=Path("/proc")):
    """Read both scopes independently; absent, denied or unsupported data stays explicit."""
    group = _cgroup(proc) if sys.platform.startswith("linux") else None
    paths = {
        "host": proc / "pressure/cpu" if sys.platform.startswith("linux") else None,
        "cgroup": group / "cpu.pressure" if group else None,
        "cpu.stat": group / "cpu.stat" if group else None,
    }
    result = {}
    for name, path in paths.items():
        text = _text(path) if path is not None else None
        result[name] = {
            "path": str(path) if path is not None else UNAVAILABLE,
            "monotonic": time.monotonic(),
            "values": _stat(text) if name == "cpu.stat" else _pressure(text),
        }
    return result


def _delta(before, after, stat=False):
    if not before or not after or before["path"] != after["path"]:
        return UNAVAILABLE
    a, b = before["values"], after["values"]
    elapsed = after["monotonic"] - before["monotonic"]
    if not isinstance(a, dict) or not isinstance(b, dict) or elapsed <= 0:
        return UNAVAILABLE
    keys = a.keys() & b.keys()
    deltas = {k: b[k] - a[k] if stat else b[k]["total"] - a[k]["total"] for k in keys}
    if not deltas or any(v < 0 for v in deltas.values()):
        return UNAVAILABLE
    if stat:
        return {"elapsed_seconds": elapsed, "delta": deltas}
    fractions = {k: v / (elapsed * 1_000_000) for k, v in deltas.items()}
    return {
        "elapsed_seconds": elapsed,
        "stall_microseconds": deltas,
        "stall_fraction": fractions,
        "contended": fractions["some"] > 0.10,
    }


def interval(before, after):
    """Use cumulative microsecond counters, not the kernel's rolling averages."""
    a, b = before.get("cpu_pressure", {}), after.get("cpu_pressure", {})
    scopes = {name: _delta(a.get(name), b.get(name)) for name in ("host", "cgroup")}
    flags = [value["contended"] for value in scopes.values() if isinstance(value, dict)]
    return {
        **scopes,
        "cpu.stat": _delta(a.get("cpu.stat"), b.get("cpu.stat"), stat=True),
        "contended": True if any(flags) else (False if len(flags) == 2 else UNAVAILABLE),
        "policy": "report_only",
    }


def noise(values):
    """Inclusive IQR, with a strict ten percent reporting boundary."""
    q1, _, q3 = (
        statistics.quantiles(values, n=4, method="inclusive")
        if len(values) > 1
        else (values[0],) * 3
    )
    median = statistics.median(values)
    return {"iqr": q3 - q1, "noisy": q3 - q1 > 0.10 * median}


def report(rows):
    flags = [row.get("cpu_pressure", {}).get("contended", UNAVAILABLE) for row in rows]
    return {
        "policy": "report_only",
        "operations": len(flags),
        "contended_operations": sum(flag is True for flag in flags),
        "unavailable_operations": flags.count(UNAVAILABLE),
    }
