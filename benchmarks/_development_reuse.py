"""Bind reused controls to complete results and the exact current measurement context."""

from __future__ import annotations

import time
from pathlib import Path

from _development_system import machine
from _measurement import document, sha256


def context(cell, prepared, identities):
    here = Path(__file__).parent
    return {
        "cell": cell,
        "models": prepared["models"][cell["model"]],
        "corpus": prepared["corpora"][cell["corpus"]],
        "identities": identities,
        "machine": machine(),
        "runner": {
            str(p.relative_to(here)): sha256(p)
            for p in sorted(here.rglob("*.py"))
            if "baseline" not in p.parts
        },
    }


def load_control(root, cell_id, suffix, expected, functional):
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
    value = document(root / name)
    if value["context"] != expected or value["functional_only"] != functional:
        raise ValueError(
            "reused control does not match the exact current cell, build, machine or runner"
        )
    return value
