"""Shared admission rules for explicitly local macOS timing diagnostics."""

from __future__ import annotations

import os
import subprocess

MAX_LOAD = 4
MAX_FOREIGN_CPU = 50


def foreign_cpu():
    """Report busy foreign processes without reading or retaining command arguments."""
    rows = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,pcpu=,comm="],
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    processes = []
    for line in rows.splitlines():
        pid, parent, cpu, command = line.split(maxsplit=3)
        processes.append(
            {
                "pid": int(pid),
                "parent": int(parent),
                "cpu_percent": float(cpu),
                "executable": command,
            }
        )
    owned = {os.getpid()}
    while True:
        descendants = {p["pid"] for p in processes if p["parent"] in owned}
        if descendants <= owned:
            break
        owned.update(descendants)
    return [p for p in processes if p["pid"] not in owned and p["cpu_percent"] > MAX_FOREIGN_CPU]
