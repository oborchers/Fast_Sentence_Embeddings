"""Private collection verification; no remote lifecycle or credentials."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from _measurement import document, sha256


def collect(root, destination):
    root = root.resolve(strict=True)
    if destination.exists() or destination.resolve().is_relative_to(root):
        raise ValueError("collection archive must be fresh and outside the result directory")
    done = document(root / "DONE")
    if sha256(root / "manifest.json") != done["manifest_sha256"]:
        raise ValueError("DONE and manifest differ")
    files = document(root / "manifest.json")["files"]
    expected = {row["path"] for row in files} | {"manifest.json", "DONE"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("result inventory differs from the manifest or contains symlinks")
    for row in files:
        p = root / row["path"]
        if not p.resolve().is_relative_to(root):
            raise ValueError("unsafe collection path")
        if p.stat().st_size != row["bytes"] or sha256(p) != row["sha256"]:
            raise ValueError(f"collection member differs: {row['path']}")
    with tarfile.open(destination, "x:gz") as stream:
        for name in sorted(expected):
            stream.add(root / name, arcname=name, recursive=False)
    receipt = {"archive": str(destination), "bytes": destination.stat().st_size,
               "sha256": sha256(destination), "members": len(expected),
               "measurement_exit_status": done["exit_status"]}
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
