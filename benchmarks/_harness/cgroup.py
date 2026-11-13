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


def _number(value, name, minimum=0, *, unlimited=False):
    if value is None or (unlimited and value == "max"):
        return value
    if not value.isascii() or not value.isdecimal() or int(value) < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _cpu_values(quota, period, *, v1=False):
    if v1 and quota is not None and quota.startswith("-"):
        _number(quota[1:], "cpu quota magnitude", 1)
        quota = "max"
    _number(quota, "cpu quota", 1, unlimited=True)
    _number(period, "cpu period", 1)
    return [quota, period] if quota is not None and period is not None else None


def _cpu(root, version):
    if version == "v2":
        value = text(root / "cpu.max")
        if value is None:
            return None
        fields = value.split()
        if len(fields) != 2:
            raise ValueError("cpu.max must contain quota and period")
        return _cpu_values(*fields)
    for folder in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
        quota = text(root / folder / "cpu.cfs_quota_us")
        period = text(root / folder / "cpu.cfs_period_us")
        values = _cpu_values(quota, period, v1=True)
        if values is not None:
            return values
    return None


def _memory_limits(memory, version):
    names = (
        ("memory.limit_in_bytes", "memory.usage_in_bytes")
        if version == "v1"
        else ("memory.max", "memory.current")
    )
    maximum, current = (text(memory / name) for name in names)
    _number(maximum, names[0], 1, unlimited=version != "v1")
    _number(current, names[1])
    # v1 uses a page-aligned LONG_MAX sentinel rather than the v2 string 'max'.
    if (
        version == "v1"
        and maximum is not None
        and int(maximum) >= sys.maxsize - (os.sysconf("SC_PAGE_SIZE") - 1)
    ):
        maximum = "max"
    return maximum, current


def _memory_stat(memory, version):
    raw = text(memory / "memory.stat")
    stat = dict(line.split() for line in raw.splitlines()) if raw is not None else {}
    for key, value in stat.items():
        _number(value, "memory.stat." + key)
    sources = dict(zip(STAT_KEYS, STAT_KEYS, strict=True))
    if version == "v1":
        # total_* includes descendants, matching the hierarchical v2 counters.
        sources = {key: "total_" + source for key, source in zip(STAT_KEYS, V1_KEYS, strict=True)}
    values = {key: int(stat[source]) if source in stat else None for key, source in sources.items()}
    return values, sources


def _missing(result):
    missing = [key for key, value in result.items() if value is None]
    missing += [
        "memory.stat." + key for key, value in result["memory.stat"].items() if value is None
    ]
    return missing


def observe(root=ROOT):
    """Keep existing v2 field names, with version and source keys for their interpretation."""
    version = _version(root)
    memory = root / "memory" if version == "v1" else root
    maximum, current = _memory_limits(memory, version)
    values, sources = _memory_stat(memory, version)
    result = {
        "cgroup_version": version,
        "memory.max": maximum,
        "memory.current": current,
        "memory.stat": values,
        "memory_stat_sources": sources if version != "unknown" else dict.fromkeys(STAT_KEYS),
        "cpu_max_raw": _cpu(root, version),
        "cgroup_io": text(root / "io.stat") if version == "v2" else None,
    }
    result["cgroup_missing_fields"] = _missing(result)
    return result


def finite_memory_limit(root=ROOT):
    value = observe(root)["memory.max"]
    if value is None or value == "max" or int(value) <= 0:
        raise ValueError("finite cgroup memory limit required")
    return int(value)
