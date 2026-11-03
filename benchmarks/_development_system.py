"""Unix capabilities and observations for non-publishable development measurements."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

LABEL = {"mode": "local-development", "publishable": False}


def source_identity():
    """Observe Git when available; file hashes remain authoritative without Git metadata."""
    unavailable = {"available": False, "revision": None, "tracked_working_tree_dirty": None}
    git = shutil.which("git")
    if not git:
        return unavailable
    root = Path(__file__).resolve().parents[1]
    with suppress(OSError):
        command = [git, "--no-optional-locks", "-C", str(root)]
        revision = subprocess.run(
            [*command, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            [*command, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=False,
        )
        if revision.returncode == 0 and status.returncode == 0:
            return {
                "available": True,
                "revision": revision.stdout.strip(),
                "tracked_working_tree_dirty": bool(status.stdout),
            }
    return unavailable


def require_unix():
    if os.name != "posix":
        raise RuntimeError("local development measurements require Unix; Windows is unsupported")


def _sysctl(name):
    with suppress(OSError):
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return None


def _optional_text(path):
    with suppress(OSError):
        return Path(path).read_text().strip()
    return None


def _affinity():
    if hasattr(os, "sched_getaffinity"):
        with suppress(OSError):
            return sorted(os.sched_getaffinity(0))
    return None


def _physical_memory():
    candidates = []
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        if value:
            candidates.append(int(value))
    elif hasattr(os, "sysconf"):
        with suppress(ValueError, OSError):
            candidates.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    return candidates


def memory_limit():
    """Return detected physical/cgroup RAM, or None rather than inventing a limit."""
    candidates = _physical_memory()
    if sys.platform.startswith("linux"):
        candidates.extend(_cgroup_memory())
    return min((n for n in candidates if n > 0), default=None)


def _cgroup_memory():
    candidates = []
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        value = _optional_text(name)
        if value and value.isdecimal():
            candidates.append(int(value))
    return candidates


def machine():
    require_unix()
    cpu = platform.processor() or platform.machine()
    physical = None
    if sys.platform == "darwin":
        cpu = _sysctl("machdep.cpu.brand_string") or cpu
        physical = _sysctl("hw.physicalcpu")
    elif sys.platform.startswith("linux"):
        fields = (_optional_text("/proc/cpuinfo") or "").splitlines()
        names = [line.split(":", 1)[1].strip() for line in fields if line.startswith("model name")]
        cpu = names[0] if names else cpu
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {
        **LABEL,
        "platform": platform.platform(),
        "system": platform.system(),
        "hostname": platform.node(),
        "boot_id": _sysctl("kern.bootsessionuuid")
        if sys.platform == "darwin"
        else _optional_text("/proc/sys/kernel/random/boot_id"),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "physical_cpus": int(physical) if physical else None,
        "memory_limit_bytes": memory_limit(),
        "affinity": _affinity(),
        "governor": _optional_text(governor) if sys.platform.startswith("linux") else None,
        "frequency_policy": "observe only; no frequency or governor changes",
    }


def foreign_cpu(excluded=()):
    """Read executable names, never process arguments, and exclude our descendants."""
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,pcpu=,comm="],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError:
        return {"available": False, "processes": []}
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


def configure_child(workers):
    require_unix()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        os.environ[name] = str(workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if workers > 1 else "false"
