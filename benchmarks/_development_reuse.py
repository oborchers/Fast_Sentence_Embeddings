"""Bind reused controls to complete results and the exact current measurement context."""

from __future__ import annotations

import json
import time
from pathlib import Path

from _development_system import machine, source_identity
from _measurement import document, sha256


def context(cell, prepared, identities):
    here = Path(__file__).parent
    return {
        "cell": cell,
        "models": prepared["models"][cell["model"]],
        "corpus": prepared["corpora"][cell["corpus"]],
        "identities": identities,
        "machine": machine(),
        "runner_git": source_identity(),
        "runner": {
            str(p.relative_to(here)): sha256(p)
            for p in sorted(here.rglob("*.py"))
            if "baseline" not in p.parts
        },
    }


def load_control(root, cell_id, suffix, expected, functional):
    if not expected["machine"]["boot_id"]:
        raise ValueError("control reuse requires an observed machine boot identity; run fresh A/A")
    root = root.resolve(strict=True)
    done = document(root / "DONE")
    manifest = root / "manifest.json"
    if done["exit_status"] or sha256(manifest) != done["manifest_sha256"]:
        raise ValueError("reused development results are incomplete or failed")
    if time.time() - done["completed_at"] >= 900:
        raise ValueError(
            "measurement session interrupted for at least fifteen minutes; new A/A required"
        )
    files = document(manifest)["files"]
    for row in files:
        path = root / row["path"]
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("unsafe reused development result path")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("reused development result bytes changed")
    name = cell_id + "-" + suffix + ".json"
    if name not in {row["path"] for row in files}:
        raise ValueError("reused control is absent from its final manifest")
    if "operations.jsonl" not in {row["path"] for row in files}:
        raise ValueError("reused cell operations are absent from its final manifest")
    _cell_gap(root, cell_id)
    value = document(root / name)
    if value["context"] != expected or value["functional_only"] != functional:
        raise ValueError(
            "reused control does not match the exact current cell, build, machine or runner"
        )
    return value


def _cell_gap(root, cell_id):
    path = root / "operations.jsonl"
    if not path.is_file():
        raise ValueError("reused control has no completed cell operations")
    with path.open() as stream:
        last = max(
            (
                row["completed_at_unix"]
                for row in map(json.loads, stream)
                if row["cell_id"] == cell_id
            ),
            default=None,
        )
    if last is None or time.time() - last >= 900:
        raise ValueError("reused cell was idle for fifteen minutes; fresh A/A required")
