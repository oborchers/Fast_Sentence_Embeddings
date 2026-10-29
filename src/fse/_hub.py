#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""A repository of the Hugging Face Hub at one commit, its files checked against it.

ARCHITECTURE.md, "from_pretrained". The extra ``fse[hub]`` (huggingface-hub) is imported
only here, on first use. ``revision`` (a branch, tag or commit; None is the default
branch) is resolved to its commit once, and every file comes from that commit. The
commit pins every file: an LFS file by its ``lfs.sha256``, any other by its git blob
sha1. A file that must be an LFS file (every safetensors file, so an encoder or a table)
without ``lfs.sha256`` raises. A file of the cache is a link of its snapshot into the
blobs of the same cached repository, which is resolved only there, and the blob is
checked. With ``local_files_only`` nothing is asked of the Hub: the commit and the files
come from the cache, and each blob is checked against its name, which is its sha256 or git
sha1 (this detects damage, not forgery).
"""

from __future__ import annotations

import io
import os
import re
import stat
from pathlib import Path
from typing import Any

from fse._container import CorruptArtifactError, file_digests, read_regular

REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")


def _library():
    """huggingface_hub, or an ImportError that names the extra."""
    try:
        import huggingface_hub  # noqa: PLC0415 - only on first use, the extra fse[hub]
    except ImportError:
        raise ImportError(
            'models of the Hugging Face Hub need huggingface-hub: pip install "fse[hub]"'
        ) from None
    return huggingface_hub


class HubFiles:
    """The files of ``repo`` at one commit (``revision``), checked when they are taken."""

    source: str | None
    revision: str | None

    def __init__(self, repo: str, *, revision, token, cache_dir, local_files_only: bool):
        if not REPO_ID.fullmatch(repo):
            raise ValueError(f"{repo!r} is no Hub repository id of the form namespace/name")
        self.hub = _library()
        self.source = repo
        self._options = {"cache_dir": cache_dir, "token": token}
        self._offline = local_files_only
        if local_files_only:
            config = self._download("config.json", revision)
            self.revision = config.parent.name
            self._pins: dict[str, tuple[str | None, str | None]] | None = None
            self._sizes: dict[str, int | None] = {}
            self._names = set(os.listdir(config.parent))
        else:
            info = self.hub.HfApi().model_info(
                repo, revision=revision, files_metadata=True, token=token
            )
            self.revision = info.sha
            self._pins = {
                s.rfilename: (getattr(s.lfs, "sha256", None) if s.lfs else None, s.blob_id)
                for s in info.siblings
            }
            # The size of an LFS file, which the commit records too.
            self._sizes = {
                s.rfilename: getattr(s.lfs, "size", None) if s.lfs else None for s in info.siblings
            }
            self._names = set(self._pins)

    def _download(self, name: str, revision: str | None) -> Path:
        path = self.hub.hf_hub_download(
            self.source,
            name,
            revision=revision,
            local_files_only=self._offline,
            **self._options,
        )
        return Path(path)

    def has(self, name: str) -> bool:
        return name in self._names

    def where(self, name: str) -> Path:
        return Path(f"{self.source}@{self.revision}") / name

    def read(
        self, name: str, *, lfs: bool = False, limit: int | None = None, inspect: Any = None
    ) -> bytes:
        """The bytes of file ``name`` of the commit, read once and checked: the bytes that
        are checked are those returned, so a blob changed afterwards changes nothing."""
        if name not in self._names:
            raise CorruptArtifactError(f"{self.where(name)} is not a file of the commit")
        path = self._download(name, self.revision)
        if path.is_symlink():
            blob, etag = resolve_link(path, path.parent.parent.parent)
        elif self._offline:
            raise CorruptArtifactError(
                f"{path} is a copy without its blob name, which local_files_only cannot check"
            )
        else:
            blob, etag = path, ""
        data = read_regular(blob, limit=limit, inspect=_sized(name, self._sizes.get(name), inspect))
        sha256, sha1 = file_digests(io.BytesIO(data), len(data))
        if self._pins is None:
            _check_name(name, etag, sha256, sha1, lfs)
        else:
            _check_pin(name, self._pins[name], sha256, sha1, lfs, self.revision)
        return data


def _sized(name: str, expected: int | None, inspect: Any) -> Any:
    """``inspect`` after a check of the size of the file against the size the commit
    records for an LFS file, both before anything but the size is read."""

    def check(handle, size: int) -> None:
        if expected is not None and size != expected:
            raise CorruptArtifactError(f"{name} has {size} bytes, the commit records {expected}")
        if inspect is not None:
            inspect(handle, size)

    return check


def resolve_link(path: Path, repo: Path) -> tuple[Path, str]:
    """The regular file a snapshot link of a Hugging Face cache points to, and its blob
    name (the etag: sha256 of an LFS file, else the git blob sha1). ``path`` lies in
    ``<repo>/snapshots/<commit>/`` and must link to ``<repo>/blobs/<etag>``, the blobs of
    the same cached repository; huggingface-hub 2 makes that entry a link into the
    cache-wide store of Xet files, ``<cache>/blobs/<2 hex digits>/<64 hex digits>``, which
    is followed too. Links are compared as written, never resolved, and no directory from
    the repository down to a blob may be a link, so nothing leads out of the cache (the
    cache folder itself may be one); anything else raises."""
    # The folders are checked here and the blob is opened later by its path, so a folder
    # swapped in between could lead elsewhere. That is no gap of integrity, and no
    # descriptor-based walk is needed: the Hub reads the bytes once and compares their
    # hash with the commit (or, with local_files_only, with the etag resolved here), so
    # whatever a swap leads to gives the pinned bytes or a refusal. Local folders carry
    # no such claim; their files are the caller's.
    repo = Path(os.path.normpath(repo))
    _plain_dirs(repo, repo / "snapshots", path.parent, repo / "blobs")
    blob = _step(path)
    if blob.parent != repo / "blobs":
        raise CorruptArtifactError(
            f"{path} is a link to {blob}, outside the blobs of its Hugging Face cache"
        )
    target = blob
    if blob.is_symlink():
        target = _step(blob)
        store = repo.parent / "blobs"
        if not (_SHA256.fullmatch(target.name) and target.parent == store / target.name[:2]):
            raise CorruptArtifactError(
                f"{blob} is a link to {target}, outside the shared blobs of its cache"
            )
        _plain_dirs(store, target.parent)
    if not stat.S_ISREG(os.lstat(target).st_mode):
        raise CorruptArtifactError(f"{path} points to {target}, which is no regular file")
    return target, blob.name


def _step(link: Path) -> Path:
    """Where one link points, as a path, compared as written (not followed further)."""
    return Path(os.path.normpath(link.parent / os.readlink(link)))


def _plain_dirs(*folders: Path) -> None:
    """Every folder is a directory of its own, no link to one elsewhere."""
    for folder in folders:
        try:
            mode = os.lstat(folder).st_mode
        except FileNotFoundError:
            raise CorruptArtifactError(f"{folder} of the Hugging Face cache is missing") from None
        if not stat.S_ISDIR(mode):
            raise CorruptArtifactError(
                f"{folder} of the Hugging Face cache is a link or no directory; fse follows "
                "no link out of the cache"
            )


def _check_pin(name: str, pin: Any, sha256: str, sha1: str, lfs: bool, commit: str | None) -> None:
    """The file against the commit: its lfs.sha256, or its git blob sha1."""
    lfs_sha256, blob_id = pin
    if lfs and lfs_sha256 is None:
        raise CorruptArtifactError(
            f"{name} of commit {commit} is no LFS file with an lfs.sha256; fse reads "
            "safetensors files only as LFS files, whose sha256 the commit pins"
        )
    if lfs_sha256 is not None:
        if sha256 != lfs_sha256:
            raise CorruptArtifactError(f"{name} has sha256 {sha256}, the commit pins {lfs_sha256}")
    elif blob_id is None or sha1 != blob_id:
        raise CorruptArtifactError(f"{name} has the git blob id {sha1}, the commit pins {blob_id}")


def _check_name(name: str, etag: str, sha256: str, sha1: str, lfs: bool) -> None:
    """A cached blob against its name in the repository: sha256 (LFS) or git blob sha1."""
    if _SHA256.fullmatch(etag):
        if sha256 != etag:
            raise CorruptArtifactError(f"{name}: the blob {etag} has sha256 {sha256}")
        return
    if lfs or not _SHA1.fullmatch(etag):
        raise CorruptArtifactError(
            f"{name}: the blob {etag} is no {'LFS ' if lfs else ''}file of the cache "
            "named by its hash"
        )
    if sha1 != etag:
        raise CorruptArtifactError(f"{name}: the blob {etag} has the git blob id {sha1}")
