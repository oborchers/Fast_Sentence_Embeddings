#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""A repository of the Hugging Face Hub at one commit, its files checked against it.

docs/architecture/hub.md, "Model2Vec models and from_pretrained". huggingface-hub, a dependency of fse, is imported only
here, on first use, so ``import fse`` and local models never load it. ``revision`` (a branch, tag or commit; None is the default
branch) is resolved to its commit once, and every file comes from that commit. The
commit pins every file: an LFS file by its ``lfs.sha256``, any other by its git blob
sha1. A file that must be an LFS file (every safetensors file, so an encoder or a table)
without ``lfs.sha256`` raises. A file of the cache is a link of its snapshot into the
blobs of the same cached repository, which is resolved only there, and the blob is
checked; a file may lie in a folder of the repository (``0_StaticEmbedding/``), and no
folder from the repository down to it may be a link. With ``local_files_only`` nothing is
asked of the Hub: the commit is the cached snapshot of ``revision`` (a commit hash, or the
commit in ``refs/<revision>``, found in the layout of the cache without
``snapshot_download``, which refuses a snapshot that lacks any file its cached tree
listing names) and the files are those below it, and each blob is checked against its name, which is its sha256 or git sha1
(this detects damage, not forgery).
"""

from __future__ import annotations

import os
import re
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from fse import _log
from fse._container import CorruptArtifactError, pin_digest, read_regular

REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_COMMIT = _SHA1  # a commit hash is a git sha1


def _library():
    """huggingface_hub, or an ImportError that names its cause."""
    try:
        import huggingface_hub  # noqa: PLC0415 - only on first use, so `import fse` stays fast
    except ImportError as error:
        raise ImportError(
            f"huggingface-hub is a dependency of fse but cannot be imported: {error}"
        ) from error
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
        self._ref: str | None = None  # the name to record as refs/<name> once a file is checked
        if local_files_only:
            snapshot = cached_snapshot(self.hub, repo, revision, cache_dir)
            self.revision = snapshot.name
            self._pins: dict[str, tuple[str | None, str | None]] | None = None
            self._sizes: dict[str, int | None] = {}
            self._names = files_below(snapshot)
        else:
            self._resolve(repo, revision, token)
        # The commit, never the repository id or revision name (a private repository's
        # name may be confidential).
        _log.emit(
            "hub.resolved",
            "fse hub: resolved a repository to commit %s (%d files%s)",
            self.revision,
            len(self._names),
            ", offline" if local_files_only else "",
            commit=self.revision,
            offline=local_files_only,
            files=len(self._names),
        )

    def _resolve(self, repo: str, revision, token) -> None:
        """Online: ``revision`` to its commit and the files with their pins."""
        name = _revision_name(revision)
        self._ref = None if _COMMIT.fullmatch(name) else name
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

    def _record_ref(self, path: Path, name: str) -> None:
        """``refs/<revision>`` = the commit this load resolved, once a file of it is checked
        (DR-184). ``hf_hub_download`` writes the ref only for a revision given by name, and
        fse downloads at the commit it resolved, so without this a later load with
        ``local_files_only`` and no revision finds no ``refs/main``. Never written offline
        or for a commit hash."""
        storage = Path(os.path.normpath(snapshot_of(path, name).parent.parent))
        if self._ref is None or self.revision is None:
            raise AssertionError("no ref to record")  # callers check; also narrows the types
        write_ref(storage, self._ref, self.revision)
        self._ref = None  # once per load

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
            blob, etag = resolve_link(path, name)
        elif self._offline:
            raise CorruptArtifactError(
                f"{path} is a copy without its blob name, which local_files_only cannot check"
            )
        else:
            snapshot_of(path, name)  # a copy, which the hash checks; its folders are not links
            blob, etag = path, ""
        data = read_regular(blob, limit=limit, inspect=_sized(name, self._sizes.get(name), inspect))
        if self._pins is None:
            pin = _check_name(name, etag, data, lfs)
        else:
            pin = _check_pin(name, self._pins[name], data, lfs, self.revision)
            if self._ref is not None:
                self._record_ref(path, name)
        _log.emit(
            "hub.file",
            "fse hub: took a checked file of %d bytes at commit %s",
            len(data),
            self.revision,
            commit=self.revision,
            bytes=len(data),
            pin=pin,
        )
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


def _revision_name(revision: str | None) -> str:
    """``revision`` (None is ``main``) when it can be a name below ``refs/``, else a
    ``ValueError``."""
    revision = "main" if revision is None else revision
    parts = revision.split("/")
    if not revision or any(p in ("", ".", "..") for p in parts) or "\\" in revision:
        raise ValueError(f"{revision!r} is no revision name of the Hugging Face cache")
    return revision


def write_ref(storage: Path, revision: str, commit: str) -> None:
    """The file ``refs/<revision>`` of the cached repository ``storage`` holds ``commit``,
    written as huggingface_hub does (``_cache_commit_hash_for_specific_revision``, which
    has no public API and cannot pin a commit): the same layout, untouched when it holds
    the commit already (a read-only cache), else a temporary file renamed over it, so a
    reader never sees half a ref. A link or other kind of entry on the way raises, as for
    the reader; two writers leave the commit of the later rename."""
    parts = ["refs", *revision.split("/")]
    for i in range(1, len(parts)):
        folder = storage.joinpath(*parts[:i])
        if not os.path.lexists(folder):
            folder.mkdir(exist_ok=True)
        _plain_dirs(folder)
    ref = storage.joinpath(*parts)
    if os.path.lexists(ref):
        if not stat.S_ISREG(os.lstat(ref).st_mode):
            raise CorruptArtifactError(
                f"{ref} is a link or the wrong kind of entry; fse follows no link in the cache"
            )
        if ref.read_text(encoding="ascii", errors="replace") == commit:
            return
    tmp = ref.with_name(f"{uuid.uuid4().hex}.tmp")  # short: the name of a ref may fill a file name
    with open(tmp, "x", encoding="ascii") as handle:  # exclusive: no other writer, no link
        handle.write(commit)
    os.replace(tmp, ref)


def cached_snapshot(hub: Any, repo: str, revision: str | None, cache_dir) -> Path:
    """The cached snapshot folder ``<cache>/models--<namespace>--<name>/snapshots/<commit>``
    of ``revision`` (None is ``main``), resolved from the documented layout of the cache:
    a commit hash names its snapshot, any other revision is the commit in the file
    ``refs/<revision>``. Nothing else is read, so files of the commit that were never
    downloaded (the tree listing that ``snapshot_download`` checks, 25 evaluation files of
    ``minishlab/potion-multilingual-128M``) do not matter: ``HubFiles.read`` takes only
    the files below the snapshot that the model needs, each checked by its blob name.
    A missing repository, ref or snapshot raises ``FileNotFoundError``, as
    ``snapshot_download(local_files_only=True)`` does (``LocalEntryNotFoundError``)."""
    revision = _revision_name(revision)
    # As huggingface_hub does: ~ is expanded and the folder resolved (it may be a link).
    base = (
        Path(hub.constants.HF_HUB_CACHE if cache_dir is None else cache_dir).expanduser().resolve()
    )
    storage = base / ("models--" + repo.replace("/", "--"))
    commit = revision
    if not _COMMIT.fullmatch(revision):
        commit = _read_ref(storage, revision, repo)
    snapshot = storage / "snapshots" / commit
    if not os.path.lexists(snapshot):
        raise FileNotFoundError(
            f"{repo}: the cache {base} has no snapshot of commit {commit} (revision "
            f"{revision!r}); local_files_only cannot load it, download it once first"
        )
    _plain_dirs(storage, storage / "snapshots", snapshot)
    return snapshot


def _read_ref(storage: Path, revision: str, repo: str) -> str:
    """The commit that the file ``refs/<revision>`` of the cached repository ``storage``
    holds; ``refs`` and the folders of the revision are plain directories, the file is
    a regular file, no link."""
    parts = ["refs", *revision.split("/")]
    for i in range(1, len(parts) + 1):
        ref = storage.joinpath(*parts[:i])
        try:
            mode = os.lstat(ref).st_mode
        except (FileNotFoundError, NotADirectoryError):
            raise FileNotFoundError(
                f"{repo}: the cache has no ref {revision!r}; local_files_only cannot resolve "
                "a revision it never downloaded (a commit hash needs no ref)"
            ) from None
        if not (stat.S_ISREG(mode) if i == len(parts) else stat.S_ISDIR(mode)):
            raise CorruptArtifactError(
                f"{ref} is a link or the wrong kind of entry; fse follows no link in the cache"
            )
    commit = ref.read_text(encoding="ascii", errors="replace").strip()
    if not _COMMIT.fullmatch(commit):
        raise CorruptArtifactError(f"{ref} holds {commit[:60]!r}, no commit hash")
    return commit


def files_below(folder: Path) -> set[str]:
    """The names of the files below ``folder`` (relative POSIX paths, links to files
    included); a folder below it that is a link raises."""
    names = set()
    with os.scandir(folder) as entries:
        found = list(entries)
    for entry in found:
        if entry.is_dir(follow_symlinks=False):
            names |= {f"{entry.name}/{n}" for n in files_below(Path(entry.path))}
        elif entry.is_symlink() and os.path.isdir(entry.path):
            raise CorruptArtifactError(
                f"{entry.path} is a link to a folder; fse follows no link out of a model"
            )
        else:
            names.add(entry.name)
    return names


def check_folders(base: Path, name: str) -> None:
    """Every folder of ``name`` below ``base`` is a directory of its own, no link."""
    parts = PurePosixPath(name).parts
    _plain_dirs(*(base.joinpath(*parts[:i]) for i in range(1, len(parts))))


def snapshot_of(path: Path, name: str) -> Path:
    """The snapshot folder ``<repo>/snapshots/<commit>`` of ``path``, the file ``name`` of
    it; every folder from the repository down to the file is a directory of its own."""
    snapshot = path.parents[len(PurePosixPath(name).parts) - 1]
    repo = Path(os.path.normpath(snapshot.parent.parent))
    _plain_dirs(repo, repo / "snapshots", snapshot)
    check_folders(snapshot, name)
    return snapshot


def resolve_link(path: Path, name: str) -> tuple[Path, str]:
    """The regular file a snapshot link of a Hugging Face cache points to, and its blob
    name (the etag: sha256 of an LFS file, else the git blob sha1). ``path`` is the file
    ``name`` (it may lie in a folder) of ``<repo>/snapshots/<commit>/`` and must link to
    ``<repo>/blobs/<etag>``, the blobs of the same cached repository; huggingface-hub 2
    makes that entry a link into the cache-wide store of Xet files,
    ``<cache>/blobs/<2 hex digits>/<64 hex digits>``, which is followed too. Links are
    compared as written, never resolved, and no directory from the repository down to a
    blob may be a link, so nothing leads out of the cache (the cache folder itself may be
    one); anything else raises."""
    # The folders are checked here and the blob is opened later by its path, so a folder
    # swapped in between could lead elsewhere. That is no gap of integrity, and no
    # descriptor-based walk is needed: the Hub reads the bytes once and compares their
    # hash with the commit (or, with local_files_only, with the etag resolved here), so
    # whatever a swap leads to gives the pinned bytes or a refusal. Local folders carry
    # no such claim; their files are the caller's.
    repo = Path(os.path.normpath(snapshot_of(path, name).parent.parent))
    _plain_dirs(repo / "blobs")
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
            raise CorruptArtifactError(f"{folder} is missing") from None
        if not stat.S_ISDIR(mode):
            raise CorruptArtifactError(
                f"{folder} is a link or no directory; fse follows no link out of a model "
                "or the Hugging Face cache"
            )


def _check_pin(name: str, pin: Any, data: bytes, lfs: bool, commit: str | None) -> str:
    """The file against the commit, by the one digest the commit pins it with: its
    lfs.sha256, or its git blob sha1. Returns the pin (``lfs_sha256`` or ``git_sha1``)."""
    lfs_sha256, blob_id = pin
    if lfs and lfs_sha256 is None:
        raise CorruptArtifactError(
            f"{name} of commit {commit} is no LFS file with an lfs.sha256; fse reads "
            "safetensors files only as LFS files, whose sha256 the commit pins"
        )
    if lfs_sha256 is not None:
        sha256 = pin_digest(data, "sha256")
        if sha256 != lfs_sha256:
            raise CorruptArtifactError(f"{name} has sha256 {sha256}, the commit pins {lfs_sha256}")
        return "lfs_sha256"
    sha1 = pin_digest(data, "git_sha1")  # also with no blob id, for the message
    if blob_id is None or sha1 != blob_id:
        raise CorruptArtifactError(f"{name} has the git blob id {sha1}, the commit pins {blob_id}")
    return "git_sha1"


def _check_name(name: str, etag: str, data: bytes, lfs: bool) -> str:
    """A cached blob against its name in the repository, by the one digest the name says:
    sha256 (LFS) or git blob sha1. Returns the pin (``lfs_sha256`` or ``git_sha1``)."""
    if _SHA256.fullmatch(etag):
        sha256 = pin_digest(data, "sha256")
        if sha256 != etag:
            raise CorruptArtifactError(f"{name}: the blob {etag} has sha256 {sha256}")
        return "lfs_sha256"
    if lfs or not _SHA1.fullmatch(etag):
        raise CorruptArtifactError(
            f"{name}: the blob {etag} is no {'LFS ' if lfs else ''}file of the cache "
            "named by its hash"
        )
    sha1 = pin_digest(data, "git_sha1")
    if sha1 != etag:
        raise CorruptArtifactError(f"{name}: the blob {etag} has the git blob id {sha1}")
    return "git_sha1"
