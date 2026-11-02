"""Content identities for benchmark inputs, prepared heads and executed runtime."""

from __future__ import annotations

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


