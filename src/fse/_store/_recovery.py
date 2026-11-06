"""Checked snapshots of immutable store shards, including unsynced publications."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy import ndarray

from fse._container import CorruptArtifactError, checksum

from ._kind import SHARD_PREFIX, Kind, Shard, shard_name
from ._kinds import kind_of
from ._layout import open_shard

NO_SHARD: object = object()  # the summary "before" the first shard


@dataclass(frozen=True)
class Snapshot:
    """Mapped rows and the verification work needed to trust this snapshot."""

    shards: list[tuple[Shard, dict[str, ndarray]]]
    unsynced_shards: int
    verified_bytes: int
    verify_seconds: float


def snapshot(path: Path, info: Any, *, verify: bool) -> Snapshot:
    """Follow the published chain, check gaps, and optionally verify tagged shards."""
    return extend(path, info, [], verify=verify)


def extend(path: Path, info: Any, old: list, *, verify: bool) -> Snapshot:
    """The shards published after the chain ``old`` (empty: from row 0), checked as
    ``snapshot`` does, with the summary of the last old shard as the one before them; the
    result holds the new shards only. Gaps and orphans are checked over old and new."""
    try:
        return _extend(path, info, old, verify=verify)
    except BaseException as error:
        clear_exception_frames(error)
        raise


def _extend(path: Path, info: Any, old: list, *, verify: bool) -> Snapshot:
    kind = kind_of(info)
    found: list = []
    previous: object = kind.summary(*old[-1]) if old else NO_SHARD
    row = old[-1][0].first_row + old[-1][0].rows if old else 0
    unsynced = verified_bytes = 0
    seconds = 0.0
    while _present(path / shard_name(row)):
        try:
            shard, tensors = open_shard(path, info, row)
            if shard.unsynced:
                unsynced += 1
                if verify:
                    start = time.perf_counter()
                    verify_shard(kind, shard, tensors)
                    verified_bytes += sum(tensor.nbytes for tensor in tensors.values())
                    seconds += time.perf_counter() - start
            current = kind.summary(shard, tensors)
            reason = chain_breach(kind, previous, current)
        except CorruptArtifactError as error:
            raise damaged(path, shard_name(row), row, str(error)) from error
        if reason is not None:
            raise damaged(path, shard_name(row), row, reason)
        previous = current
        found.append((shard, tensors))
        row += shard.rows
    check_gaps(path, [*old, *found], row)
    return Snapshot(found, unsynced, verified_bytes, seconds)


def rows_in(shards: list) -> int:
    """The rows of the shards of a snapshot."""
    return sum(shard.rows for shard, _ in shards)


def stop_row(shards: list, start: int, stop: int | None) -> int:
    """The end of the rows ``start`` to ``stop - 1`` of the snapshot ``shards`` (all rows
    from ``start`` if ``stop`` is None); they must lie within it."""
    n_rows = rows_in(shards)
    stop = n_rows if stop is None else stop
    if not start <= stop <= n_rows:
        raise ValueError(f"rows {start} to {stop} are not within the {n_rows} rows")
    return stop


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
            "are not; append with durability='none' or write a new store"
        )


def verify_shard(kind: Kind, shard: Shard, tensors: dict[str, ndarray]) -> None:
    """Hash the mapped tensors against the checksums from that same manifest open, then
    check the values of the shard."""
    for name, tensor in tensors.items():
        if checksum(tensor) != shard.checksums[name]:
            raise CorruptArtifactError(f"{shard.name}: the checksum of {name} does not match")
    kind.check_values(shard, tensors)


def chain_breach(kind: Kind, previous: object, current: object) -> str | None:
    """Why a shard with the summary ``current`` cannot follow the one before it, which has
    the summary ``previous`` (None for the first shard of the store), or None."""
    return None if previous is NO_SHARD else kind.follows(previous, current)


def check_chain(path: Path, kind: Kind, found: list) -> None:
    """Check every shard against the one before it, with the same ``follows`` for snapshot,
    ``verify`` and repair; a breach is the damage of the later shard."""
    previous: object = NO_SHARD
    for shard, tensors in found:
        current = kind.summary(shard, tensors)
        reason = chain_breach(kind, previous, current)
        if reason is not None:
            raise damaged(path, shard.name, shard.first_row, reason)
        previous = current


def clear_exception_frames(error: BaseException) -> None:
    """Release unreturned maps held in completed frames, retaining the traceback.

    Numpy has no public memmap close API. Clearing completed frames releases
    their array references without invalidating any live array or shared mapping.
    """
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(
            cause for cause in (current.__cause__, current.__context__) if cause is not None
        )
