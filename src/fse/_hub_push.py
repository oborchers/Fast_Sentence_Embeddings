#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""A model folder pushed to the Hugging Face Hub in exactly one commit, checked after it.

ARCHITECTURE.md, "Hub folders". The repository is read at one commit; every file of the
folder is compared with the file of the same name there by its digest (``lfs.sha256`` of
an LFS file, else the git blob id): an equal one is left as it is, a different one is
refused unless ``replace``. License files already there (LICENSE, NOTICE, COPYING) that
the folder does not bring are kept, ``.gitattributes`` always; any other file is refused
unless ``replace``, which deletes exactly those names. The operations go into one
``create_commit`` on top of that commit (``parent_commit``), so a concurrent push fails
instead of mixing, and no upload is split into several commits (``upload_folder`` with
hf-xet commits large folders in batches, and its ``delete_patterns`` are patterns). At
the returned commit, the file set and every digest are checked.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fse import _hub
from fse._container import CorruptArtifactError, file_digests
from fse._container._common import _open_regular

REPO_ID = _hub.REPO_ID
# License texts a repository keeps when a push does not bring its own.
LICENSE_FILE = re.compile(r"(?i)(licen[cs]e|notice|copying)([._-][A-Za-z0-9._-]*)?")
_KEPT = ".gitattributes"


def local_digests(folder: Path) -> dict[str, tuple[str, str]]:
    """The sha256 and git blob id of every file of the flat ``folder``."""
    out = {}
    for name in sorted(os.listdir(folder)):
        with _open_regular(folder / name) as handle:
            out[name] = file_digests(handle, os.fstat(handle.fileno()).st_size)
    return out


def _remote(info: Any) -> dict[str, tuple[str | None, str | None]]:
    """The files of a commit: name to (lfs.sha256 or None, git blob id)."""
    return {
        s.rfilename: (getattr(s.lfs, "sha256", None) if s.lfs else None, s.blob_id)
        for s in info.siblings
    }


def _same(name: str, local: tuple[str, str], remote: tuple[str | None, str | None]) -> bool:
    """The remote file is the local one as fse reads it: a ``.safetensors`` file only as
    an LFS file with its sha256, any other by its sha256 or git blob id."""
    sha256, sha1 = local
    lfs, blob = remote
    if lfs is None and name.endswith(".safetensors"):
        return False  # stored as a plain git blob, which fse refuses to read
    return lfs == sha256 if lfs is not None else blob == sha1


def plan(
    local: dict[str, tuple[str, str]],
    remote: dict[str, tuple[str | None, str | None]],
    replace: bool,
) -> tuple[list[str], list[str], set[str]]:
    """The names to add, the names to delete and the names the commit keeps."""
    differ = sorted(n for n in local if n in remote and not _same(n, local[n], remote[n]))
    foreign = sorted(n for n in remote if n not in local and not _keeps(n))
    if not replace and (differ or foreign):
        raise ValueError(_refusal(differ, foreign, _plain(differ, remote)))
    add = sorted(n for n in local if n not in remote or n in differ)
    kept = {n for n in remote if n not in local and n not in foreign}
    return add, foreign, kept


def _keeps(name: str) -> bool:
    """A file of the repository that a push leaves as it is: .gitattributes, licenses."""
    return name == _KEPT or LICENSE_FILE.fullmatch(name) is not None


def _plain(names: list[str], remote: dict[str, tuple[str | None, str | None]]) -> list[str]:
    """The ``.safetensors`` files of ``names`` that the repository holds as plain blobs."""
    return [n for n in names if n.endswith(".safetensors") and remote[n][0] is None]


def _refusal(differ: list[str], foreign: list[str], plain: list[str]) -> str:
    parts = [f"other content under {differ}"] if differ else []
    if plain:
        parts[0] += f" ({plain} stored as plain git blobs, no LFS files that fse reads)"
    parts += [f"files of another model {foreign}"] if foreign else []
    return (
        f"the repository holds {' and '.join(parts)}; pass replace=True to replace and remove "
        "them in the same commit"
    )


def push_folder(
    folder: Path,
    repo_id: str,
    *,
    token: Any,
    private: bool | None,
    revision: str | None,
    create_pr: bool,
    commit_message: str,
    replace: bool,
) -> str:
    """Push the files of ``folder`` to ``repo_id`` in one commit; returns its URL."""
    hub = _hub._library()
    local = local_digests(folder)
    api = hub.HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    parent, remote = info.sha, _remote(info)
    add, delete, kept = plan(local, remote, replace)
    expected: dict[str, tuple[Any, Any]] = {n: remote[n] for n in kept}
    expected.update(local)  # (sha256, git blob id) of every file of the folder
    if not add and not delete:  # the commit holds the folder already: checked as it is
        check_files(remote, expected, f"commit {parent} of {repo_id}")
        return f"{hub.constants.ENDPOINT}/{repo_id}/commit/{parent}"
    operations = [hub.CommitOperationDelete(path_in_repo=n, is_folder=False) for n in delete]
    operations += [
        hub.CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(folder / n)) for n in add
    ]
    description = "".join(f"Removes {n}.\n" for n in delete)
    description += "".join(f"Replaces {n}.\n" for n in add if n in remote)
    commit = api.create_commit(
        repo_id,
        operations,
        commit_message=commit_message,
        commit_description=description or None,
        repo_type="model",
        revision=revision,
        create_pr=create_pr,
        parent_commit=parent,
    )
    check_commit(api, repo_id, commit.oid, expected)
    return commit.pr_url if create_pr and commit.pr_url else commit.commit_url


def check_commit(api: Any, repo_id: str, commit: str, expected: dict[str, tuple[Any, Any]]) -> None:
    """The commit holds exactly the expected files, each with its digest; every
    ``.safetensors`` file is an LFS file. The commit is not undone on a mismatch."""
    found = _remote(api.model_info(repo_id, revision=commit, files_metadata=True))
    check_files(found, expected, f"commit {commit} of {repo_id}")


def check_files(
    found: dict[str, tuple[str | None, str | None]],
    expected: dict[str, tuple[Any, Any]],
    where: str,
) -> None:
    """``found`` (the files of a commit) holds exactly the expected files with their
    digests, and every ``.safetensors`` file is an LFS file."""
    if set(found) != set(expected):
        missing, extra = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
        raise CorruptArtifactError(
            f"{where} lacks {missing} and holds {extra}; the commit stays as it is"
        )
    for name, want in expected.items():
        lfs, blob = found[name]
        if name.endswith(".safetensors") and lfs is None:
            raise CorruptArtifactError(
                f"{name} of {where} is no LFS file with an lfs.sha256, which fse needs to read "
                "it; the commit stays as it is"
            )
        # An LFS file by its sha256 (the local one, or the kept file's), else the git blob id.
        if (want[0] != lfs) if lfs is not None else (want[1] != blob):
            raise CorruptArtifactError(
                f"{name} of {where} differs from the file that was pushed; the commit stays as it is"
            )
