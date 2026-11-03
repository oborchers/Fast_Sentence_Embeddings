"""Unix capabilities and observations for non-publishable development measurements."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

LABEL = {"mode": "local-development", "publishable": False}


def require_unix():
    if os.name != "posix":
        raise RuntimeError("local development measurements require Unix; Windows is unsupported")


def _sysctl(name):
    result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def memory_limit():
    """Return detected physical/cgroup RAM, or None rather than inventing a limit."""
    candidates = []
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        if value:
            candidates.append(int(value))
    elif hasattr(os, "sysconf"):
        try:
            candidates.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass
    if sys.platform.startswith("linux"):
        for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            path = Path(name)
            if path.is_file():
                value = path.read_text().strip()
                if value.isdecimal():
                    candidates.append(int(value))
    return min((n for n in candidates if n > 0), default=None)


def machine():
    require_unix()
    cpu = platform.processor() or platform.machine()
    physical = None
    if sys.platform == "darwin":
        cpu = _sysctl("machdep.cpu.brand_string") or cpu
        physical = _sysctl("hw.physicalcpu")
    elif sys.platform.startswith("linux") and Path("/proc/cpuinfo").is_file():
        fields = Path("/proc/cpuinfo").read_text().splitlines()
        names = [line.split(":", 1)[1].strip() for line in fields if line.startswith("model name")]
        cpu = names[0] if names else cpu
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {
        **LABEL,
        "platform": platform.platform(),
        "system": platform.system(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "physical_cpus": int(physical) if physical else None,
        "memory_limit_bytes": memory_limit(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "governor": governor.read_text().strip()
        if sys.platform.startswith("linux") and governor.is_file()
        else None,
        "frequency_policy": "observe only; no frequency or governor changes",
    }


def foreign_cpu(excluded=()):
    """Read executable names, never process arguments, and exclude our descendants."""
    result = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,pcpu=,comm="],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode:
        return {"available": False, "processes": []}
    rows = []
    for line in result.stdout.splitlines():
        pid, parent, cpu, executable = line.split(maxsplit=3)
        rows.append(
            dict(pid=int(pid), parent=int(parent), cpu_percent=float(cpu), executable=executable)
        )
    owned = {os.getpid(), *excluded}
    while True:
        descendants = {row["pid"] for row in rows if row["parent"] in owned}
        if descendants <= owned:
            break
        owned.update(descendants)
    return {
        "available": True,
        "policy": "telemetry_only",
        "processes": [r for r in rows if r["pid"] not in owned and r["cpu_percent"] > 50],
    }


def observations(work, excluded=()):
    require_unix()
    import resource

    return {
        **LABEL,
        "unix": time.time(),
        "monotonic": time.monotonic(),
        "load": list(os.getloadavg()),
        "nice": os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else None,
        "foreign_cpu": foreign_cpu(excluded),
        "disk": shutil.disk_usage(work)._asdict(),
        "usage": list(resource.getrusage(resource.RUSAGE_SELF)),
    }
