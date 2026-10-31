"""Checked snapshots of immutable store shards, including unsynced publications."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray

from fse._container import CorruptArtifactError, checksum

from ._layout import SHARD_PREFIX, Shard, StoreInfo, open_shard, shard_name


@dataclass(frozen=True)
class Snapshot:
    """Mapped rows and the verification work needed to trust this snapshot."""

    shards: list[tuple[Shard, dict[str, ndarray]]]
    unsynced_shards: int
    verified_bytes: int
    verify_seconds: float


def snapshot(path: Path, info: StoreInfo, *, verify: bool) -> Snapshot:
    """Follow the published chain, check gaps, and optionally verify tagged shards."""
    found = []
    row = unsynced = verified_bytes = 0
    seconds = 0.0
    while _present(path / shard_name(row)):
        try:
            shard, tensors = open_shard(path, info, row)
            if shard.unsynced:
                unsynced += 1
                if verify:
                    start = time.perf_counter()
                    verify_shard(shard, tensors)
                    verified_bytes += sum(tensor.nbytes for tensor in tensors.values())
                    seconds += time.perf_counter() - start
        except CorruptArtifactError as error:
            raise damaged(path, shard_name(row), row, str(error)) from error
        found.append((shard, tensors))
        row += shard.rows
    check_gaps(path, found, row)
    return Snapshot(found, unsynced, verified_bytes, seconds)


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def canonical_shards(path: Path) -> list[tuple[int, str]]:
    """Canonical shard entries only; dot entries and unrelated names are ignored."""
    found = []
    for entry in path.iterdir():
        suffix = entry.name.removeprefix(SHARD_PREFIX)
        if not entry.name.startswith(SHARD_PREFIX) or not suffix.isdecimal():
            continue
        row = int(suffix)
        if entry.name == shard_name(row):
            found.append((row, entry.name))
    return sorted(found)


def check_gaps(path: Path, found: list, end: int) -> None:
    """Reject orphans, while retaining a snapshot if a writer has since extended it."""
    members = {shard.name for shard, _ in found}
    outside = [(row, name) for row, name in canonical_shards(path) if name not in members]
    for row, name in outside:
        if row < end:
            raise damaged(path, name, row, "a shard overlaps the published chain")
    if outside and not _present(path / shard_name(end)):
        row, name = outside[0]
        raise damaged(path, name, row, f"a shard follows a gap from row {end}")


def damaged(path: Path, name: str, row: int, reason: str) -> CorruptArtifactError:
    """Corruption always names the affected shard, row and explicit recovery action."""
    return CorruptArtifactError(
        f"{path / name}, first row {row}: {reason}; run fse.repair_store(path) "
        "while no writer or reader runs"
    )


def refuse_unsynced(path: Path, found: list) -> None:
    """A full writer cannot promise durability behind an unsynced prefix."""
    first = next((shard.first_row for shard, _ in found if shard.unsynced), None)
    if first is not None:
        raise ValueError(
            f"the store {path} holds rows written with durability='none' from row {first}; "
            "rows appended with durability='full' cannot be durable while rows before them "
            "are not; append with durability='none', or write a new store"
        )


def verify_shard(shard: Shard, tensors: dict[str, ndarray]) -> None:
    """Hash the mapped tensors against the checksums from that same manifest open."""
    for name, tensor in tensors.items():
        if checksum(tensor) != shard.checksums[name]:
            raise CorruptArtifactError(f"{shard.name}: the checksum of {name} does not match")
    check_values(shard, tensors)


def check_values(shard: Shard, tensors: dict[str, ndarray]) -> None:
    """Reject non-finite values, negative inverse norms/scales and invalid int8 codes."""
    for name in ("inv_norms", "scales"):
        values = tensors.get(name)
        if values is not None and not (np.isfinite(values) & (values >= 0)).all():
            raise CorruptArtifactError(
                f"{shard.name}: {name} holds values that are not finite and at least 0"
            )
    vectors = tensors["vectors"]
    if vectors.dtype == np.int8:
        if (vectors == -128).any():
            raise CorruptArtifactError(f"{shard.name}: an int8 code is -128, beyond +-127")
    elif not np.isfinite(vectors).all():
        raise CorruptArtifactError(f"{shard.name}: the vectors hold values that are not finite")
