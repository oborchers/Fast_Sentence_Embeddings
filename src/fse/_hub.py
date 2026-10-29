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

import os
import re
import stat
from pathlib import Path
from typing import Any

from fse._container import CorruptArtifactError, file_digests

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

    def get(self, name: str, *, lfs: bool = False) -> Path | None:
        """The checked blob of file ``name`` of the commit, or None if it has none."""
        if name not in self._names:
            return None
        path = self._download(name, self.revision)
        if path.is_symlink():
            blob = resolve_link(path, path.parent.parent.parent / "blobs")
        elif self._offline:
            raise CorruptArtifactError(
                f"{path} is a copy without its blob name, which local_files_only cannot check"
            )
        else:
            blob = path
        sha256, sha1 = _digests(blob)
        if self._pins is None:
            _check_name(name, blob, sha256, sha1, lfs)
        else:
            _check_pin(name, self._pins[name], sha256, sha1, lfs, self.revision)
        return blob


def resolve_link(path: Path, blobs: Path) -> Path:
    """The file a snapshot link of a Hugging Face cache points to: a regular file in
    ``blobs``, the blobs of the same cached repository, never anything else."""
    target = Path(os.path.realpath(path))
    if target.parent != Path(os.path.realpath(blobs)):
        raise CorruptArtifactError(
            f"{path} is a link to {target}, outside the blobs of its Hugging Face cache"
        )
    if not stat.S_ISREG(os.lstat(target).st_mode):
        raise CorruptArtifactError(f"{path} points to {target}, which is no regular file")
    return target


def _digests(path: Path) -> tuple[str, str]:
    with open(path, "rb") as handle:
        return file_digests(handle, os.fstat(handle.fileno()).st_size)


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


def _check_name(name: str, blob: Path, sha256: str, sha1: str, lfs: bool) -> None:
    """A cached blob against its name: its sha256 (LFS) or git blob sha1."""
    if _SHA256.fullmatch(blob.name):
        if sha256 != blob.name:
            raise CorruptArtifactError(f"{name}: the blob {blob.name} has sha256 {sha256}")
        return
    if lfs or not _SHA1.fullmatch(blob.name):
        raise CorruptArtifactError(
            f"{name}: the blob {blob.name} is no {'LFS ' if lfs else ''}file of the cache "
            "named by its hash"
        )
    if sha1 != blob.name:
        raise CorruptArtifactError(f"{name}: the blob {blob.name} has the git blob id {sha1}")
