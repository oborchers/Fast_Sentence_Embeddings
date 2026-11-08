"""Private provenance, telemetry and environment records shared by both measurement entries."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

from _harness.records import append, sha256
from _measurement_config import CONFIG


def spread(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite observations required")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def package_identity(name):
    from importlib import import_module, metadata

    module = import_module(name.replace("-", "_"))
    distribution = metadata.distribution(name)
    digest = hashlib.sha256()
    if not distribution.files:
        raise ValueError("backend distribution has no verifiable installed files")
    root = Path(module.__file__).resolve().parent
    files = {str(item): Path(distribution.locate_file(item)) for item in distribution.files}
    files.update({"loaded/" + str(p.relative_to(root)): p for p in root.rglob("*")})
    for relative, path in sorted(files.items()):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(relative.encode() + b"\0" + bytes.fromhex(sha256(path)))
    return {
        "package": name,
        "version": distribution.version,
        "path": module.__file__,
        "sha256": digest.hexdigest(),
    }


def telemetry():
    root = Path("/sys/fs/cgroup")
    work = Path(os.getenv("MEASURE_WORK", "."))
    disk = shutil.disk_usage(work if work.exists() else Path.cwd())
    result = {
        "unix": time.time(),
        "monotonic": time.monotonic(),
        "load": os.getloadavg(),
        "disk": disk._asdict(),
    }
    for name in ("memory.max", "memory.current"):
        p = root / name
        result[name] = p.read_text().strip() if p.exists() else None
    stat = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
    result["memory.stat"] = {
        key: int(stat[key])
        for key in ("anon", "file", "file_dirty", "file_writeback", "inactive_file")
    }
    result["meminfo"] = [
        line
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("Dirty:", "Writeback:", "MemAvailable:"))
    ]
    result["process_io"] = Path("/proc/self/io").read_text()
    io = root / "io.stat"
    result["cgroup_io"] = io.read_text() if io.exists() else None
    return result


def rehearsal_limits():
    """The rehearsal machine rules of config/allocation.json (the same check an allocation
    runs before it starts a step); the observed limits of the container."""
    from _harness import machine, spec

    expected = spec.load(CONFIG / "allocation.json", "speed")["rehearsal_machine_check"]
    checked = machine.check(expected, Path(os.getenv("MEASURE_WORK", ".")))
    if checked["status"] != "accepted":
        raise ValueError("rehearsal machine check rejected: " + "; ".join(checked["deviations"]))
    os.environ["CARGO_BUILD_JOBS"] = "2"
    observed = checked["observed"]
    quota, period = observed["cpu_max_raw"]
    return {
        "memory_bytes": observed["memory_max_bytes"],
        "cpu_quota": int(quota),
        "cpu_period": int(period),
        "cargo_jobs": 2,
    }


def write_environment(args, contract):
    atomic(
        args.results / "environment.json",
        {
            "contract": contract,
            "measurement_source_sha256": {
                str(p.relative_to(Path(__file__).resolve().parent)): sha256(p)
                for p in sorted(Path(__file__).resolve().parent.rglob("*.py"))
                if "baseline" not in p.parts
            },
            "platform": platform.platform(),
            "python": sys.version,
            "telemetry": telemetry(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "machine": {
                "cpu": next(
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "affinity": sorted(os.sched_getaffinity(0)),
                "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
                "allocation_id": os.getenv("SPEED_ALLOCATION_ID"),
                "allocated_at_unix": float(
                    os.getenv("SPEED_MACHINE_ALLOCATED_AT", args.allocated_at)
                ),
                "budget_started_at_unix": args.allocated_at,
            },
        },
    )


def finalize_rehearsal(ledger, entry):
    if not ledger.contract.get("functional_only"):
        return
    from _measurement_rehearse import control_proof

    try:
        control_proof(ledger.root / "controls", entry)
    except Exception as error:
        append(ledger.root / "fatal.jsonl", {"stage": "rehearsal-controls", "error": repr(error)})
