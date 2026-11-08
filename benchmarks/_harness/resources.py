"""One memory and disk plan for every study: components against the real limit.

Every study sizes its work in components (stores, model copies, widened weights, head
vectors, quality files, profiles, disk peaks) and admits it against the memory the process
may really use and the disk that is free. The limit is the smallest of the cgroup limit
(``memory.max``, or the version 1 file), the physical RAM and an explicit request, and the
plan names which one it was. The production plan of a study (full size) and the plan of a
functional run (reduced input) are separate records of the same shape: ``kind`` tells which.
Standard library only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

GIB = 1024**3
SAFETY = 0.85  # the share of the limit a plan may use
KINDS = ("production", "functional")
CGROUP_FILES = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")


def _cgroup():
    for name in CGROUP_FILES:
        with suppress(OSError):
            value = Path(name).read_text().strip()
            if value.isdecimal() and int(value) > 0:
                return int(value)
    return None


def _physical():
    if sys.platform == "darwin":
        with suppress(OSError, ValueError):
            done = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
            )
            return int(done.stdout.strip()) if done.returncode == 0 else None
    with suppress(ValueError, OSError):
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return None


def memory_limit(requested=None):
    """``(bytes, source)`` of the smallest limit that applies: ``cgroup``, ``physical`` or
    ``requested``; ``(None, None)`` when nothing is detectable and nothing is requested."""
    found = {"cgroup": _cgroup(), "physical": _physical(), "requested": requested}
    known = {name: value for name, value in found.items() if value and value > 0}
    if not known:
        return None, None
    source = min(known, key=known.get)
    return known[source], source


def require_limit(requested=None):
    limit, source = memory_limit(requested)
    if not limit:
        raise ValueError("RAM cannot be detected; supply --memory-limit-bytes")
    return limit, source


def assess(
    components=None,
    *,
    phases=None,
    limit,
    work=None,
    disk=None,
    kind="production",
    disk_fraction=1.0,
    **extra,
):
    """The plan of one workload. The peak is the sum of ``components`` or, where the parts run
    one after another, the largest of ``phases``; it is admitted when it fits ``SAFETY`` of
    the limit and ``disk`` (when given, with a ``work`` folder) fits the free space
    (``disk_fraction`` of it). ``extra`` fields are
    carried unchanged."""
    if kind not in KINDS or (components is None) == (phases is None):
        raise ValueError(f"a plan has a kind in {KINDS} and components or phases")
    bytes_, source = limit
    peak = sum(components.values()) if components is not None else max(phases.values())
    free = None if work is None else shutil.disk_usage(work).free
    fits_disk = disk is None or (free is not None and free * disk_fraction >= disk)
    return {
        "kind": kind,
        **extra,
        **({"components": components} if components is not None else {"phases": phases}),
        "estimated_peak_bytes": peak,
        "memory_limit_bytes": bytes_,
        "limit_source": source,
        "safety_fraction": SAFETY,
        "required_disk_bytes": disk,
        "available_disk_bytes": free,
        "admitted": peak <= bytes_ * SAFETY and fits_disk,
    }
