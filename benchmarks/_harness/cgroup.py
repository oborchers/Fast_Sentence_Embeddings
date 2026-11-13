"""Container-visible cgroup v1/v2 observations, shared by admission and telemetry.

The injected root is for functional probes only. Missing files stay explicit; malformed
numeric data raises before setup when the machine check calls telemetry.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path("/sys/fs/cgroup")
STAT_KEYS = ("anon", "file", "file_dirty", "file_writeback", "inactive_file")
V1_KEYS = ("rss", "cache", "dirty", "writeback", "inactive_file")


def text(path):
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def _version(root):
    if (root / "cgroup.controllers").exists() or (root / "memory.max").exists():
        return "v2"
    if any(
        (root / "memory" / name).exists()
        for name in ("memory.stat", "memory.limit_in_bytes", "memory.usage_in_bytes")
    ):
        return "v1"
    return "unknown"


def _cpu(root, version):
    if version == "v2":
        value = text(root / "cpu.max")
        return value.split() if value is not None else None
    for folder in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
        quota = text(root / folder / "cpu.cfs_quota_us")
        period = text(root / folder / "cpu.cfs_period_us")
        if quota is not None and period is not None:
            return ["max" if int(quota) < 0 else quota, period]
    return None


def observe(root=ROOT):
    """Keep existing v2 field names, with version and source keys for their interpretation."""
    version = _version(root)
    memory = root / "memory" if version == "v1" else root
    names = (
        ("memory.limit_in_bytes", "memory.usage_in_bytes")
        if version == "v1"
        else ("memory.max", "memory.current")
    )
    maximum, current = (text(memory / name) for name in names)
    # v1 uses a page-aligned LONG_MAX sentinel rather than the v2 string 'max'.
    if version == "v1" and maximum is not None and int(maximum) >= sys.maxsize - (os.sysconf("SC_PAGE_SIZE") - 1):
        maximum = "max"
    raw = text(memory / "memory.stat")
    stat = dict(line.split() for line in raw.splitlines()) if raw is not None else {}
    sources = dict(zip(STAT_KEYS, STAT_KEYS, strict=True))
    if version == "v1":
        # total_* includes descendants, matching the hierarchical v2 counters.
        sources = {key: "total_" + source for key, source in zip(STAT_KEYS, V1_KEYS, strict=True)}
    values = {key: int(stat[source]) if source in stat else None for key, source in sources.items()}
    result = {
        "cgroup_version": version,
        "memory.max": maximum,
        "memory.current": current,
        "memory.stat": values,
        "memory_stat_sources": sources if version != "unknown" else dict.fromkeys(STAT_KEYS),
        "cpu_max_raw": _cpu(root, version),
        "cgroup_io": text(root / "io.stat") if version == "v2" else None,
    }
    result["cgroup_missing_fields"] = [key for key, value in result.items() if value is None]
    result["cgroup_missing_fields"] += [
        "memory.stat." + key for key, value in values.items() if value is None
    ]
    return result


def finite_memory_limit(root=ROOT):
    value = observe(root)["memory.max"]
    if value is None or value == "max" or int(value) <= 0:
        raise ValueError("finite cgroup memory limit required")
    return int(value)
