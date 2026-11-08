"""The machine check before any measurement starts.

An allocation declares what the machine must be. The check observes the machine once and
compares: any deviation, and any declared value it cannot observe, rejects the machine and
no step starts. Standard library only; it runs on a bare interpreter right after the clone.
"""

from __future__ import annotations

import os
import resource
import shutil
import time
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")
SHM = Path("/dev") / "shm"
RANGES = ("affinity_cpus", "cpu_quota_cpus", "memory_max_bytes", "disk_free_bytes", "shm_bytes")
FIELDS = ("cpu_model", "required_flags", "nofile_hard_min", *RANGES)


def _cpuinfo():
    path = Path("/proc/cpuinfo")
    lines = path.read_text().splitlines() if path.exists() else []
    model = next((x.split(":", 1)[1].strip() for x in lines if x.startswith("model name")), None)
    flags = next((x for x in lines if x.startswith(("flags", "Features"))), "")
    return model, sorted(flags.split(":", 1)[1].split()) if ":" in flags else []


def _cgroup(name):
    path = CGROUP / name
    return path.read_text().split() if path.exists() else None


def observe(work):
    """What the machine is now. None marks a value the machine does not offer."""
    model, flags = _cpuinfo()
    cpu_max, memory_max = _cgroup("cpu.max"), _cgroup("memory.max")
    quota = None if not cpu_max or cpu_max[0] == "max" else int(cpu_max[0]) / int(cpu_max[1])
    memory = None if not memory_max or memory_max[0] == "max" else int(memory_max[0])
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu_model": model,
        "flags": flags,
        "affinity": affinity,
        "affinity_cpus": None if affinity is None else len(affinity),
        "cpu_max_raw": cpu_max,
        "cpu_quota_cpus": quota,
        "memory_max_raw": memory_max,
        "memory_max_bytes": memory,
        "disk_free_bytes": shutil.disk_usage(work).free,
        "shm_bytes": shutil.disk_usage(SHM).total if SHM.exists() else None,
        "nofile_soft": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        "nofile_hard": resource.getrlimit(resource.RLIMIT_NOFILE)[1],
    }


def validate(expected):
    unknown = set(expected) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown machine check fields: {sorted(unknown)}")
    for name in RANGES:
        bounds = expected.get(name, {})
        if set(bounds) - {"min", "max"}:
            raise ValueError(f"{name} takes only min and max")
    return expected


def _range(name, value, bounds, unlimited_ok):
    """Deviations of one number against its bounds. An unlimited quota or memory (None)
    satisfies a minimum and violates a maximum; any other missing value deviates."""
    if value is None:
        if unlimited_ok and "max" not in bounds:
            return []
        return [f"{name}: not observable or unlimited, declared {bounds}"]
    low, high = bounds.get("min"), bounds.get("max")
    if (low is not None and value < low) or (high is not None and value > high):
        return [f"{name}: observed {value}, declared {bounds}"]
    return []


def deviations(observed, expected):
    found = []
    model = expected.get("cpu_model")
    if model and model != observed["cpu_model"]:
        found.append(f"cpu_model: observed {observed['cpu_model']!r}, declared {model!r}")
    missing = sorted(set(expected.get("required_flags", [])) - set(observed["flags"]))
    if missing:
        found.append(f"flags: missing {missing}")
    floor = expected.get("nofile_hard_min")
    if floor is not None and observed["nofile_hard"] < floor:
        found.append(f"nofile_hard: observed {observed['nofile_hard']}, declared minimum {floor}")
    for name in RANGES:
        if name in expected:
            unlimited = name in ("cpu_quota_cpus", "memory_max_bytes")
            found += _range(name, observed[name], expected[name], unlimited)
    return found


def check(expected, work):
    """The record of one check: accepted only without any deviation."""
    observed = observe(work)
    found = deviations(observed, validate(expected))
    return {
        "status": "rejected" if found else "accepted",
        "checked_at": time.time(),
        "expected": expected,
        "observed": observed,
        "deviations": found,
    }
