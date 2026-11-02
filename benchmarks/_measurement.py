"""Private durable records, provenance and admission shared by both measurement entries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def document(path):
    return json.loads(path.read_text())


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("x") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append(path, value):
    line = json.dumps(value, allow_nan=False, separators=(",", ":"))
    with path.open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


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


def admit(contract, allocated_at, estimate):
    billing = contract["billing"]
    hourly = billing["compute_usd_per_hour"] + (
        contract["machine"]["disk_gb"] * billing["disk_usd_per_gb_month"] / billing["month_hours"]
    )
    ceiling = min(billing["planning_minutes"] * 60, billing["planning_balance_usd"] * 3600 / hourly)
    return (
        max(0, time.time() - allocated_at)
        + estimate
        + billing.get("collection_reserve_seconds", 600)
        <= ceiling
    )


def rehearsal_limits():
    root = Path("/sys/fs/cgroup")
    memory = (root / "memory.max").read_text().strip()
    quota, period = (root / "cpu.max").read_text().split()
    if memory == "max" or int(memory) > 4 * 1024**3:
        raise ValueError("rehearsal requires a cgroup memory limit of at most four GiB")
    if quota == "max" or int(quota) > 2 * int(period):
        raise ValueError("rehearsal requires a cgroup CPU quota of at most two CPUs")
    os.environ["CARGO_BUILD_JOBS"] = "2"
    return {
        "memory_bytes": int(memory),
        "cpu_quota": int(quota),
        "cpu_period": int(period),
        "cargo_jobs": 2,
    }


class Ledger:
    def __init__(self, root, contract, allocated_at):
        self.root, self.contract, self.allocated_at = root, contract, allocated_at
        self.finished, self.stopping = [], False
        self.current, self.stage = "preparation", "setup"
        self.lock = threading.Lock()

    def admit(self, estimate):
        return not self.stopping and admit(self.contract, self.allocated_at, estimate)

    def progress(self, terminal=False):
        with self.lock:
            remaining = [
                c
                for c in self.contract["cells"]
                if c["id"] not in {r["cell_id"] for r in self.finished}
            ]
            scale = (
                statistics.median(
                    [
                        r["duration_seconds"] / r["estimated_seconds"]
                        for r in self.finished
                        if r["status"] == "valid"
                    ]
                )
                if any(r["status"] == "valid" for r in self.finished)
                else 1
            )
            atomic(
                self.root / "progress.json",
                {
                    "stage": "collection" if terminal else self.stage,
                    "cell": "DONE" if terminal else self.current,
                    "finished": len(self.finished),
                    "planned": len(self.contract["cells"]),
                    "elapsed_seconds": time.time() - self.allocated_at,
                    "estimated_remaining_seconds": sum(c["estimated_seconds"] for c in remaining)
                    * scale,
                },
            )

    def result(self, cell, started, **result):
        row = {
            **cell,
            "cell_id": cell["id"],
            "run_id": self.contract["format"],
            "machine_environment_sha256": sha256(self.root / "environment.json"),
            "platform": platform.platform(),
            "python": sys.version,
            "completed_at_unix": time.time(),
            "date_utc": datetime.now(UTC).isoformat(),
            "machine": document(self.root / "environment.json")["machine"],
            "duration_seconds": time.monotonic() - started,
            "functional_only": self.contract.get("functional_only", False),
            **result,
        }
        append(self.root / "cells.jsonl", row)
        self.finished.append(row)
        self.progress()

    def done(self):
        self.progress(terminal=True)
        files = [
            {
                "path": str(p.relative_to(self.root)),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", "DONE")
        ]
        atomic(self.root / "manifest.json", {"files": files})
        code = int(
            len(self.finished) != len(self.contract["cells"])
            or any(r["status"] != "valid" for r in self.finished)
            or (self.root / "fatal.jsonl").exists()
        )
        atomic(
            self.root / "DONE",
            {
                "exit_status": code,
                "completed_at": time.time(),
                "cells": len(self.finished),
                "manifest_sha256": sha256(self.root / "manifest.json"),
            },
        )
        return code


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
