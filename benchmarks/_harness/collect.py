"""Collection of finished result folders into verified archives, for every kind of result.

A folder is collectable when DONE names the manifest hash, the manifest lists exactly the
files present, and every file matches its size and hash. Standard library only; no remote
lifecycle and no credentials.
"""

from __future__ import annotations

import json
import tarfile

from _harness.records import SEALED, document, sha256


def verify(root):
    """Check a sealed folder against its manifest; the verified file list and DONE."""
    root = root.resolve(strict=True)
    if not (root / SEALED[1]).is_file() or not (root / SEALED[0]).is_file():
        raise ValueError(f"{root} is not sealed: manifest.json and DONE required")
    done = document(root / SEALED[1])
    if sha256(root / SEALED[0]) != done["manifest_sha256"]:
        raise ValueError("DONE and manifest differ")
    files = document(root / SEALED[0])["files"]
    expected = {row["path"] for row in files} | set(SEALED)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("result inventory differs from the manifest or contains symlinks")
    for row in files:
        p = root / row["path"]
        if not p.resolve().is_relative_to(root):
            raise ValueError("unsafe collection path")
        if p.stat().st_size != row["bytes"] or sha256(p) != row["sha256"]:
            raise ValueError(f"collection member differs: {row['path']}")
    return expected, done


def collect(root, destination):
    root = root.resolve(strict=True)
    if destination.exists() or destination.resolve().is_relative_to(root):
        raise ValueError("collection archive must be fresh and outside the result directory")
    expected, done = verify(root)
    with tarfile.open(destination, "x:gz") as stream:
        for name in sorted(expected):
            stream.add(root / name, arcname=name, recursive=False)
    receipt = {
        "archive": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "members": len(expected),
        "measurement_exit_status": done["exit_status"],
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt
