"""Restore one hash-checked raw evidence file from the separately fetched archive."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

MANIFEST = Path(__file__).resolve().parents[1] / "baseline/dr151_evidence_manifest_20261101.json"


def restore(manifest, name, output):
    record = json.loads(manifest.read_text())
    commit = record["evidence_commit"]
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("evidence commit must be a full object ID")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence path must be relative and cannot escape its tree")
    matches = [row for row in record["files"] if row["path"] == name]
    if len(matches) != 1:
        raise ValueError("evidence path must identify exactly one manifest entry")
    done = subprocess.run(["git", "show", commit + ":" + name], capture_output=True, check=False)
    if done.returncode:
        raise RuntimeError("fetch benchmark-evidence-2.0 before restoring its pinned files")
    expected = matches[0]
    if (
        len(done.stdout) != expected["size_bytes"]
        or hashlib.sha256(done.stdout).hexdigest() != expected["sha256"]
    ):
        raise ValueError("archived evidence differs from its size or SHA256 receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(done.stdout)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.manifest, args.path, args.output)))


if __name__ == "__main__":
    main()
