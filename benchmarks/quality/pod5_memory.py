"""Fail-closed Linux cgroup v2 evidence for the explicitly supported pod layout.

Kernel reference: https://docs.kernel.org/admin-guide/cgroup-v2.html
A private cgroup namespace must expose this process's own cgroup at the mount root.
Visible ancestors are rejected rather than assuming their limits do not apply.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def snapshot(root=Path("/sys/fs/cgroup"), proc=Path("/proc"), pid=None):
    pid = os.getpid() if pid is None else pid
    membership = (proc / "self/cgroup").read_text().splitlines()
    if membership != ["0::/"]:
        raise RuntimeError("Pod 5 requires a private cgroup v2 namespace rooted at its own cgroup")
    mounts = [line.split() for line in (proc / "self/mountinfo").read_text().splitlines()]
    matches = [
        fields
        for fields in mounts
        if fields[4] == str(root) and fields[fields.index("-") + 1] == "cgroup2"
    ]
    if len(matches) != 1:
        raise RuntimeError("Pod 5 requires one cgroup v2 mount at /sys/fs/cgroup")
    if str(pid) not in (root / "cgroup.procs").read_text().splitlines():
        raise RuntimeError("cgroup mount root does not contain the recording process")
    values = {}
    for name in ("memory.max", "memory.current", "memory.peak"):
        value = (root / name).read_text().strip()
        if not value.isdecimal():
            raise RuntimeError(f"Pod 5 needs a finite numeric {name} counter")
        values[name] = int(value)
    if values["memory.max"] <= 0 or values["memory.peak"] < values["memory.current"]:
        raise RuntimeError("invalid cgroup memory counters")
    return {
        **values,
        "cgroup_namespace": "private root",
        "mount_root": matches[0][3],
        "ancestor_scope": "namespace root limit; host ancestors require coordinator allocation verification",
    }


def preflight():
    state = snapshot()
    print(json.dumps(state), flush=True)
    if state["memory.max"] < 58 * 2**30:
        raise RuntimeError("Pod 5 requires at least 58 GiB effective cgroup RAM")
    if state["memory.max"] - state["memory.current"] < 56 * 2**30:
        raise RuntimeError("Pod 5 needs at least 56 GiB cgroup headroom before setup")
