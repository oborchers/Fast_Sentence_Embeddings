"""Durable result records: atomic writes, inventory, manifest and DONE.

Standard library only and free of imports from the rest of the package: the bootstrap script
inlines this file to seal the results of a run that never started.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SEALED = ("manifest.json", "DONE")


def document(path):
    return json.loads(path.read_text())


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:  # a leftover of a dead writer is replaced
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def append(path, value):
    line = json.dumps(value, allow_nan=False, separators=(",", ":"))
    with path.open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def inventory(root):
    """Every file of a result folder except its own manifest and DONE, with size and hash."""
    return [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p not in (root / SEALED[0], root / SEALED[1])
    ]


def exit_code(rows, planned, root):
    """Zero only when every planned cell has a valid row and no fatal record exists."""
    return int(
        len(rows) != planned
        or any(r["status"] != "valid" for r in rows)
        or (root / "fatal.jsonl").exists()
    )


def seal(root, exit_status, cells, label=None, extra=None):
    """Write manifest.json over the folder, then DONE over the manifest. DONE is last: a
    folder without it is not finished and collection refuses it."""
    label = label or {}
    atomic(root / SEALED[0], {**label, "files": inventory(root)})
    atomic(
        root / SEALED[1],
        {
            **label,
            "exit_status": exit_status,
            "completed_at": time.time(),
            "cells": cells,
            **(extra or {}),
            "manifest_sha256": sha256(root / SEALED[0]),
        },
    )


def seal_unstarted(root, exit_status, reason=None):
    """Seal a folder of a run that ended before it could seal itself (a crash, a refusal, a
    restart): an empty cell list is enough, the status and the reason carry the failure. A
    folder that is already sealed stays as it is."""
    if (root / "DONE").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    if reason:
        append(root / "fatal.jsonl", {"error": reason})
    (root / "cells.jsonl").touch(exist_ok=True)
    rows = sum(1 for _ in (root / "cells.jsonl").open())
    atomic(
        root / "progress.json", {"stage": "terminal", "finished": rows, "exit_status": exit_status}
    )
    seal(root, exit_status, rows)


def read_rows(root, name="cells.jsonl"):
    path = Path(root) / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
