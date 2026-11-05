"""Offline repair: verify all shards, then durably quarantine a damaged tail."""

from __future__ import annotations

import errno
import re
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fse import _log
from fse._container import CorruptArtifactError, NpyContainer
from fse._container._common import _strict_json, _sync_directory
from fse._container._npy import _check_version, _manifest_bytes
from fse._container._npy_inplace import _rename_exclusive
from fse._contract import UnsupportedArtifactError
from fse._validation import PathLike, validate_arguments

from ._kind import Kind, shard_name
from ._kinds import kind_of
from ._layout import open_shard, read_store
from ._recovery import (
    NO_SHARD,
    canonical_shards,
    chain_breach,
    clear_exception_frames,
    verify_shard,
)


@dataclass(frozen=True)
class _Checked:
    first_row: int
    rows: int | None
    good: bool
    summary: object = None

    @property
    def name(self) -> str:
        return shard_name(self.first_row)


@validate_arguments
def repair_store(path: PathLike) -> int:
    """Keep the verified prefix of a store and return its number of rows.

    Run only while no writer and no reader runs. Every shard is checked, including
    full-durability shards. The first bad shard, its tail and all orphans move into
    a new ``.broken-<hex>`` directory, in descending row order. Every move is synced
    before the next starts. Nothing is deleted; an interrupted repair can be rerun.
    Unknown formats raise UnsupportedArtifactError before any mutation. Durability
    tags and store.json are never changed or promoted to trusted.
    """
    start = time.perf_counter()
    root = Path(path)
    info = read_store(root)
    checked = _preflight(root, info)
    _complete_quarantines(root)
    rows, kept = _prefix(kind_of(info), checked)
    moved = [entry for row, entry in sorted(checked.items(), reverse=True) if row not in kept]
    if moved:
        _quarantine(root, moved)
    moved_rows = (
        None
        if any(entry.rows is None for entry in moved)
        else sum(entry.rows for entry in moved if entry.rows is not None)
    )
    _log.emit(
        "store.repaired",
        "fse store: kept %d rows and moved %d shards aside",
        rows,
        len(moved),
        rows=rows,
        moved_shards=len(moved),
        moved_rows=moved_rows,
        seconds=time.perf_counter() - start,
    )
    return rows


def _preflight(path: Path, info: Any) -> dict[int, _Checked]:
    # Inspect every canonical entry before mkdir/rename, even after corruption. An
    # unsupported later shard must leave the entire store untouched.
    return {row: _checked(path, info, row) for row, _ in canonical_shards(path)}


def _checked(path: Path, info: Any, row: int) -> _Checked:
    try:
        entry, error = _inspect(path, info, row)
    except BaseException as error:
        clear_exception_frames(error)
        raise
    if error is not None:
        clear_exception_frames(error)  # _inspect has returned; all its maps can go
    return entry


def _inspect(path: Path, info: Any, row: int) -> tuple[_Checked, BaseException | None]:
    try:
        _check_identifiers(path / shard_name(row), info)
        shard, tensors = open_shard(path, info, row)
    except CorruptArtifactError as error:
        cause = error.__cause__
        if isinstance(cause, OSError) and cause.errno not in (
            errno.ENOENT,
            errno.ENOTDIR,
            errno.ELOOP,
        ):
            raise cause from None
        return _Checked(row, None, False), error
    kind = kind_of(info)
    try:
        verify_shard(kind, shard, tensors)
        summary = kind.summary(shard, tensors)
    except CorruptArtifactError as error:
        return _Checked(row, shard.rows, False), error
    return _Checked(row, shard.rows, True, summary), None


def _check_identifiers(directory: Path, info: Any) -> None:
    """Recognize readable future identifiers before applying today's structure rules."""
    if not directory.is_dir() or directory.is_symlink():
        return
    file = directory / "manifest.json"
    manifest = _strict_json(_manifest_bytes(directory, file), file)
    if not isinstance(manifest, dict):
        return  # open_shard diagnoses corruption of the known format
    container = manifest.get("container")
    if isinstance(container, str) and container != NpyContainer.name:
        raise UnsupportedArtifactError(f"{file}: unsupported container {container!r}")
    if "container" in manifest and "container_version" in manifest:
        _check_version(directory, manifest)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return
    kind_of(info).check_identifiers(metadata, file, info)


def _prefix(kind: Kind, checked: dict[int, _Checked]) -> tuple[int, set[int]]:
    """The verified prefix: shards in a chain from row 0 that are good, and each follows the
    one before it. The first shard that breaks the chain is damaged, with its tail."""
    row, kept, previous = 0, set(), NO_SHARD
    while (entry := checked.get(row)) is not None and entry.good:
        assert entry.rows is not None
        if chain_breach(kind, previous, entry.summary) is not None:
            break
        kept.add(row)
        previous = entry.summary
        row += entry.rows
    return row, kept


def _quarantine(path: Path, moved: list[_Checked]) -> None:
    broken = path / f".broken-{secrets.token_hex(8)}"
    broken.mkdir()
    _sync_directory(path)
    for entry in moved:
        _rename_exclusive(path / entry.name, broken / entry.name)
        _sync_directory(broken)
        _sync_directory(path)


def _complete_quarantines(path: Path) -> None:
    """Finish a last rename whose directory syncs a prior process never reached."""
    found = False
    for entry in path.iterdir():
        if re.fullmatch(r"\.broken-[0-9a-f]{16}", entry.name) and stat.S_ISDIR(
            entry.lstat().st_mode
        ):
            _sync_directory(entry)
            found = True
    if found:
        _sync_directory(path)
