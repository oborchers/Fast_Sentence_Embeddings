"""The machine check before any measurement starts.

An allocation declares what the machine must be. The check observes the machine once and
compares: any deviation, and any declared value it cannot observe, rejects the machine and
no step starts. Standard library only; it runs on a bare interpreter right after the clone.
"""

from __future__ import annotations

import csv
import os
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path

from _harness import cgroup, host_quiet
from _harness.pressure import snapshot
from _harness.telemetry import telemetry

SHM = Path("/dev") / "shm"
RANGES = ("affinity_cpus", "cpu_quota_cpus", "memory_max_bytes", "disk_free_bytes", "shm_bytes")
GPU_FIELDS = ("gpu_name", "gpu_count", "gpu_memory_bytes", "cuda_driver_version")
FIELDS = ("cpu_model", "required_flags", "nofile_hard_min", "host_quiet", *RANGES, *GPU_FIELDS)
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def _version(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", value):
        raise ValueError("cuda_driver_version must be a dotted numeric driver version")
    parts = tuple(int(part) for part in value.split("."))
    return (*parts, 0) if len(parts) == 2 else parts


def _gpu():
    """Observe all NVIDIA devices, with memory.total converted from MiB to bytes.

    A failed or malformed query leaves every GPU field unobservable. CPU-only
    allocations do not need this query or an NVIDIA installation.
    """
    missing = dict.fromkeys((*GPU_FIELDS, "gpu_devices"))
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        devices = [_gpu_row(row) for row in csv.reader(output.splitlines(), skipinitialspace=True)]
    except (OSError, subprocess.SubprocessError, ValueError, csv.Error):
        return missing
    return {
        "gpu_devices": devices,
        "gpu_count": len(devices),
        "gpu_name": devices[0]["name"] if len(devices) == 1 else None,
        "gpu_memory_bytes": min((d["memory_bytes"] for d in devices), default=None),
        "cuda_driver_version": min(
            (d["driver_version"] for d in devices), key=_version, default=None
        ),
    }


def _gpu_row(row):
    if len(row) != 4:
        raise ValueError("GPU query needs name, memory, driver and UUID")
    name, memory, driver, uuid = (part.strip() for part in row)
    _version(driver)
    if not name or not GPU_UUID.fullmatch(uuid) or not memory.isascii() or not memory.isdecimal():
        raise ValueError("GPU query contains unavailable fields")
    if int(memory) <= 0:
        raise ValueError("GPU memory must be positive")
    return {
        "name": name,
        "memory_bytes": int(memory) * 1024**2,
        "driver_version": driver,
        "uuid": uuid,
    }


def _cpuinfo():
    path = Path("/proc/cpuinfo")
    lines = path.read_text().splitlines() if path.exists() else []
    model = next((x.split(":", 1)[1].strip() for x in lines if x.startswith("model name")), None)
    flags = next((x for x in lines if x.startswith(("flags", "Features"))), "")
    return model, sorted(flags.split(":", 1)[1].split()) if ":" in flags else []


def observe(work, *, gpu=False):
    """What the machine is now. None marks a value the machine does not offer."""
    if type(gpu) is not bool:
        raise ValueError("gpu must be a boolean")
    model, flags = _cpuinfo()
    group = cgroup.observe()
    cpu_max = group["cpu_max_raw"]
    memory_max = group["memory.max"]
    memory_max = [memory_max] if memory_max is not None else None
    quota = None if not cpu_max or cpu_max[0] == "max" else int(cpu_max[0]) / int(cpu_max[1])
    memory = None if not memory_max or memory_max[0] == "max" else int(memory_max[0])
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        **(_gpu() if gpu else {}),
        "cpu_pressure": snapshot(),
        "cgroup_version": group["cgroup_version"],
        "cgroup_missing_fields": group["cgroup_missing_fields"],
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
    if not isinstance(expected, dict):
        raise ValueError("machine check must be an object")
    unknown = set(expected) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown machine check fields: {sorted(unknown)}")
    for name in RANGES:
        bounds = expected.get(name, {})
        if set(bounds) - {"min", "max"}:
            raise ValueError(f"{name} takes only min and max")
    _validate_gpu(expected)
    if "host_quiet" in expected:
        host_quiet.validate(expected["host_quiet"])
    return expected


def _validate_gpu(expected):
    if "gpu_name" in expected:
        name = expected["gpu_name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("gpu_name must be a nonempty exact device name")
    for name in ("gpu_count", "gpu_memory_bytes"):
        if name in expected and (type(expected[name]) is not int or expected[name] <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if "cuda_driver_version" in expected:
        _version(expected["cuda_driver_version"])


def _gpu_deviations(observed, expected):
    found = []
    for name in GPU_FIELDS:
        if name not in expected:
            continue
        value, declared = observed.get(name), expected[name]
        if name == "cuda_driver_version":
            matches = value is not None and _version(value) >= _version(declared)
        elif name == "gpu_memory_bytes":
            matches = value is not None and value >= declared
        else:
            matches = value == declared
        if not matches:
            found.append(f"{name}: observed {value!r}, declared {declared!r}")
    return found


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
    return found + _gpu_deviations(observed, expected)


def check(expected, work, *, budget=None):
    """The record of one check: accepted only without any deviation."""
    validate(expected)
    initial = telemetry(work=work)
    observed = observe(work, gpu=any(name in expected for name in GPU_FIELDS))
    found = deviations(observed, expected)
    if "host_quiet" in expected:
        observed["host_quiet"] = host_quiet.observe(expected["host_quiet"], budget=budget)
        found += host_quiet.deviations(observed["host_quiet"], expected["host_quiet"])
    return {
        "status": "rejected" if found else "accepted",
        "checked_at": time.time(),
        "expected": expected,
        "observed": observed,
        "telemetry": initial,
        "deviations": found,
    }
