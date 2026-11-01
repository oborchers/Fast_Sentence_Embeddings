"""DR190-compatible, checksummed pod artifacts and complete environment records."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path


def emit(name: str, value) -> None:
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    zipped = gzip.compress(raw, mtime=0)
    encoded = base64.b64encode(zipped).decode()
    chunks = [encoded[i : i + 160] for i in range(0, len(encoded), 160)]
    print(f"DR190_BEGIN {name} {len(zipped)} {hashlib.sha256(zipped).hexdigest()}", flush=True)
    for index, chunk in enumerate(chunks):
        print(f"DR190_DATA {name} {index} {chunk}", flush=True)
    print(f"DR190_END {name} {len(chunks)}", flush=True)


def environment() -> dict:
    data = json.loads(subprocess.check_output(["/usr/bin/lscpu", "--json"], text=True))
    cpu = {row["field"].rstrip(":"): row["data"] for row in data["lscpu"]}
    return {
        "platform": platform.platform(),
        "cpu": cpu.get("Model name", platform.processor()),
        "cpu_details": cpu,
        "python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if Path("/sys/fs/cgroup/cpu.max").exists()
        else None,
        "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if Path("/sys/fs/cgroup/memory.max").exists()
        else None,
    }


def export_jsonl(name: str, path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    emit(name, {"format": "jsonl", "path": str(path), "rows": rows})
