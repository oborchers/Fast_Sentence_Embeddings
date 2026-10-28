#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The number of CPUs this process may use, the default number of workers.

ARCHITECTURE.md, "Defaults": an explicit override (``PYTHON_CPU_COUNT`` or
``-X cpu_count``, as in Python 3.13) wins. Otherwise the CPUs of the process affinity,
limited by the CPU quota of its Linux cgroup and every visible ancestor (v2 ``cpu.max``,
v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``), rounded up. A container whose host
has 128 CPUs but grants the process 4 thus gets 4. Quotas of ancestors outside the
container's view are invisible to it.
"""

from __future__ import annotations

import functools
import math
import os
import sys
from pathlib import Path

_PROC = Path("/proc/self")


def usable_cpus() -> int:
    """The CPUs this process may use, at least 1."""
    override = _override()
    if override is not None:
        return override
    # macOS and Windows have no affinity call.
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = _process_quota()
    return max(1, cpus if quota is None else min(cpus, quota))


@functools.cache
def _process_quota() -> int | None:
    """The cgroup quota of this process, read once: it costs a few microseconds per read,
    which an inline call for one sentence would pay every time."""
    return _cgroup_quota(_PROC, Path("/"))


def _override() -> int | None:
    """The CPU count given by ``-X cpu_count`` or ``PYTHON_CPU_COUNT``; "default" is none."""
    value = sys._xoptions.get("cpu_count")
    if value is None or value is True:
        value = os.environ.get("PYTHON_CPU_COUNT")
    if value is None or value == "default":
        return None
    if not isinstance(value, str) or not value.isdecimal() or int(value) < 1:
        raise ValueError(f"PYTHON_CPU_COUNT must be a positive integer or 'default', not {value!r}")
    return int(value)


def _cgroup_quota(proc: Path, root: Path) -> int | None:
    """The smallest CPU quota of the process's cgroup and its visible ancestors, rounded
    up, or None without one. ``root`` prefixes the mount points (for tests)."""
    try:
        memberships = (proc / "cgroup").read_text(encoding="utf-8").splitlines()
        mounts = (proc / "mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    quotas = []
    for version, path in _cgroups(memberships):
        mount = _mount(mounts, version)
        if mount is None:
            continue
        top = root / mount[1].lstrip("/")
        directory = _directory(mount[0], top, path)
        if directory is not None:
            quotas += _quotas(directory, top, version)
    return min(quotas, default=None)


def _cgroups(memberships: list[str]) -> list[tuple[int, str]]:
    """(version, path) of the cgroups that hold the CPU controller: v2, and v1 'cpu'."""
    found = []
    for line in memberships:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            found.append((2, path))
        elif "cpu" in controllers.split(","):
            found.append((1, path))
    return found


def _mount(mounts: list[str], version: int) -> tuple[str, str] | None:
    """(root, mount point) of the cgroup filesystem of ``version`` with the CPU controller."""
    for line in mounts:
        fields = line.split()
        if "-" not in fields:
            continue
        rest = fields[fields.index("-") + 1 :]
        if len(fields) < 5 or len(rest) < 3:
            continue
        fstype, options = rest[0], rest[2].split(",")
        if (version == 2 and fstype == "cgroup2") or (
            version == 1 and fstype == "cgroup" and "cpu" in options
        ):
            return fields[3], fields[4]
    return None


def _directory(mount_root: str, top: Path, path: str) -> Path | None:
    """The directory of cgroup ``path`` under the mount ``top`` of ``mount_root``, if the
    mount shows it."""
    relative = os.path.relpath(path, mount_root)
    if relative == ".." or relative.startswith("../"):
        return None
    return top if relative == "." else top / relative


def _quotas(directory: Path, top: Path, version: int) -> list[int]:
    """The rounded-up quotas of ``directory`` and its ancestors up to the mount ``top``."""
    quotas = []
    while True:
        quota = _quota(directory, version)
        if quota is not None:
            quotas.append(quota)
        if directory == top or top not in directory.parents:
            return quotas
        directory = directory.parent


def _quota(directory: Path, version: int) -> int | None:
    try:
        if version == 2:
            quota, period = (directory / "cpu.max").read_text(encoding="utf-8").split()[:2]
        else:
            quota = (directory / "cpu.cfs_quota_us").read_text(encoding="utf-8").strip()
            period = (directory / "cpu.cfs_period_us").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    if quota in ("max", "-1") or not quota.isdecimal() or not period.isdecimal():
        return None
    if int(period) == 0:
        return None
    return max(1, math.ceil(int(quota) / int(period)))
