"""Content identities for benchmark inputs, prepared heads and executed runtime."""

from __future__ import annotations

import json
from pathlib import Path


def file_digest(path: Path) -> str:
    from fse._container._hash import Hasher

    hasher = Hasher()
    with path.open("rb") as stream:
        while block := stream.read(8 * 2**20):
            hasher.update(block)
    return hasher.checksum()


def runtime_identity() -> str:
    import fse
    from fse._container._hash import Hasher

    root = Path(fse.__file__).resolve().parent
    hasher = Hasher()
    paths = [*root.rglob("*.py"), *root.glob("_native*.so"), *root.glob("_native*.pyd")]
    for path in sorted(paths):
        hasher.update(str(path.relative_to(root)).encode("utf-8"))
        hasher.update(file_digest(path).encode("ascii"))
    return hasher.checksum()


def seal_preparation(folder: Path) -> None:
    path = folder / "manifest.json"
    record = json.loads(path.read_text())
    record["artifacts"] = {
        file.name: {"file_bytes": file.stat().st_size, "checksum": file_digest(file)}
        for file in sorted(folder.iterdir())
        if file.is_file() and file != path
    }
    path.write_text(json.dumps(record, indent=2) + "\n")


def validate_preparation(folder: Path, revision: str, evaluation_digest: str) -> dict:
    record = json.loads((folder / "manifest.json").read_text())
    if (
        record["dataset_revision"] != revision
        or record["evaluation_digest"] != evaluation_digest
        or not record.get("artifacts")
    ):
        raise ValueError(f"{folder.name}: classifier preparation does not match this evaluation")
    return record
