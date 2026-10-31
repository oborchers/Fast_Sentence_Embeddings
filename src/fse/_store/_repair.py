"""Offline repair: verify all shards, then durably quarantine a damaged tail."""

from __future__ import annotations

import errno
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from fse import _log
from fse._container import CorruptArtifactError
from fse._container._common import _strict_json, _sync_directory
from fse._container._npy import _check_version, _manifest_bytes
from fse._container._npy_inplace import _rename_exclusive
from fse._contract import UnsupportedArtifactError
from fse._validation import PathLike, validate_arguments

from ._layout import StoreInfo, _check_quantization, open_shard, read_store, shard_name
from ._recovery import canonical_shards, clear_exception_frames, verify_shard


@dataclass(frozen=True)
class _Checked:
    first_row: int
    rows: int | None
    good: bool

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
    rows, kept = _prefix(checked)
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


def _preflight(path: Path, info: StoreInfo) -> dict[int, _Checked]:
    # Inspect every canonical entry before mkdir/rename, even after corruption. An
    # unsupported later shard must leave the entire store untouched.
    return {row: _checked(path, info, row) for row, _ in canonical_shards(path)}


def _checked(path: Path, info: StoreInfo, row: int) -> _Checked:
    try:
        entry, error = _inspect(path, info, row)
    except BaseException as error:
        clear_exception_frames(error)
        raise
    if error is not None:
        clear_exception_frames(error)  # _inspect has returned; all its maps can go
    return entry


def _inspect(path: Path, info: StoreInfo, row: int) -> tuple[_Checked, BaseException | None]:
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
    try:
        verify_shard(shard, tensors)
    except CorruptArtifactError as error:
        return _Checked(row, shard.rows, False), error
    return _Checked(row, shard.rows, True), None


def _check_identifiers(directory: Path, info: StoreInfo) -> None:
    """Recognize readable future identifiers before applying today's structure rules."""
    if not directory.is_dir() or directory.is_symlink():
        return
    file = directory / "manifest.json"
    manifest = _strict_json(_manifest_bytes(directory, file), file)
    if not isinstance(manifest, dict):
        return  # open_shard diagnoses corruption of the known format
    if "container" in manifest and "container_version" in manifest:
        _check_version(directory, manifest)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return
    durability = metadata.get("fse.durability")
    if isinstance(durability, str) and durability != "none":
        raise UnsupportedArtifactError(f"{file}: unsupported fse.durability {durability!r}")
    if isinstance(metadata.get("fse.quantization"), str):
        _check_quantization(metadata, info, file)


def _prefix(checked: dict[int, _Checked]) -> tuple[int, set[int]]:
    row, kept = 0, set()
    while (entry := checked.get(row)) is not None and entry.good:
        assert entry.rows is not None
        kept.add(row)
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
