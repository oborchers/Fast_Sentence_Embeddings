"""Immutable search inputs and executing code, checked outside every timer."""

from __future__ import annotations

from pathlib import Path

from _harness.records import sha256
from _measurement import package_identity

HERE = Path(__file__).resolve().parent


def adapters():
    paths = [HERE / "speed.py", *HERE.glob("_*.py"), *(HERE / "_harness").glob("*.py")]
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(paths)}


def store_identity(path):
    path = Path(path)
    manifests = [path / "store.json", *sorted(path.glob("*/manifest.json"))]
    if len(manifests) < 2:
        raise ValueError("search store has no shard manifests")
    return {str(p.relative_to(path)): sha256(p) for p in manifests}


def identity(job):
    return {
        "store": store_identity(job["store"]),
        "queries": sha256(Path(job["queries_file"])),
        "references": {key: sha256(Path(p)) for key, p in job["reference_files"].items()},
        "artifact": sha256(Path(job["artifact"])),
        "package": package_identity("fse"),
        "adapters": adapters(),
    }


def verify(job):
    actual = identity(job)
    if actual != job["search_identity"]:
        raise ValueError("search input, reference, store or executing code changed")
    return actual


def require_bits(store):
    if not store._shards or any(shard.sign_bits is None for shard, _ in store._shards):
        raise ValueError("binary search requires sign bits for every shard")
