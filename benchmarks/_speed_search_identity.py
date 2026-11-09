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
    if not (path / "store.json").is_file() or not list(path.glob("shard-*/manifest.json")):
        raise ValueError("search store has no shard manifests")
    members = sorted(path.rglob("*"))
    if path.is_symlink() or any(p.is_symlink() for p in members):
        raise ValueError("search store contains a symlink")
    # Bind payloads too, including each sign-bit companion. A changed vector outside
    # the checked top-k must fail even after the last measured round.
    return {str(p.relative_to(path)): sha256(p) for p in members if p.is_file()}


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
