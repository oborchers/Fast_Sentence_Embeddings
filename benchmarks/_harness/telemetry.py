"""Standard-library telemetry, usable by bare-interpreter admission and both entries."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from _harness import cgroup
from _harness.pressure import snapshot


def telemetry(root=cgroup.ROOT, proc=Path("/proc"), work=None):
    work = work if work is not None else Path(os.getenv("MEASURE_WORK", "."))
    disk = shutil.disk_usage(work if work.exists() else Path.cwd())
    result = {
        "unix": time.time(),
        "monotonic": time.monotonic(),
        "load": os.getloadavg(),
        "cpu_pressure": snapshot(proc),
        **cgroup.observe(root),
        "disk": disk._asdict(),
    }
    meminfo = cgroup.text(proc / "meminfo")
    result["meminfo"] = (
        [
            line
            for line in meminfo.splitlines()
            if line.startswith(("Dirty:", "Writeback:", "MemAvailable:"))
        ]
        if meminfo is not None
        else None
    )
    result["process_io"] = cgroup.text(proc / "self/io")
    result["missing_fields"] = result["cgroup_missing_fields"] + [
        key for key in ("meminfo", "process_io") if result[key] is None
    ]
    return result
