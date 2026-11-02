"""Explicit publication and anonymous readback of a frozen payload manifest."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from payload import check
from verify import runtime

from fse._container._common import _sync_directory


def write_receipt(path, value):
    """Publish evidence before another remote operation or local cleanup."""
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def revision_info(api, entry, revision):
    info = api.model_info(entry["repo"], revision=revision, files_metadata=True)
    if info.id != entry["repo"] or info.private is not False:
        raise ValueError("public repository identity or visibility differs")
    if not re.fullmatch(r"[a-f0-9]{40}", info.sha or ""):
        raise ValueError("Hub returned no exact revision")
    if {item.rfilename for item in info.siblings} != set(entry["files"]):
        raise ValueError("remote inventory differs")
    return info.sha


def readback(entry, scratch, revision="main"):
    """Download an exact public revision into a fresh cache and check every byte."""
    revision = revision_info(HfApi(token=False), entry, revision)
    # Only the cache created here is cleaned, never caller-owned payloads or caches.
    with tempfile.TemporaryDirectory(prefix="hub-readback-", dir=scratch) as temporary:
        folder = Path(
            snapshot_download(
                entry["repo"], revision=revision, token=False, cache_dir=temporary, max_workers=2
            )
        )
        # Hub snapshots contain symlinks. Verify resolved regular payloads in our cache.
        materialized = Path(temporary) / "payload"
        materialized.mkdir()
        for name in entry["files"]:
            source = (folder / name).resolve(strict=True)
            if not source.is_relative_to(Path(temporary).resolve()):
                raise ValueError("download escaped the fresh cache")
            os.link(source, materialized / name)
        files = check(materialized, entry)
        probe = runtime(materialized, entry)
    return {
        "repo": entry["repo"],
        "revision": revision,
        "files": files,
        "runtime": probe,
        "status": "passed",
    }


def require_absent(api, repo):
    try:
        api.repo_info(repo, repo_type="model")
    except RepositoryNotFoundError as error:
        if getattr(error.response, "status_code", None) == 404:
            return
        raise
    raise FileExistsError(f"repository already exists: {repo}")


def upload(entries, models, scratch, receipt, enabled):
    """New public repositories only. Never repair or overwrite an existing name."""
    if not enabled:
        raise ValueError("Hub writes require the explicit --upload switch")
    if receipt.exists():
        raise FileExistsError("publication receipt exists; inspect it before any retry")
    api = HfApi()
    api.whoami()
    for entry in entries:
        check(models / entry["repo"].split("/")[1], entry)
        require_absent(api, entry["repo"])
    state = {"status": "running", "repositories": [], "hub_writes_performed": False}
    write_receipt(receipt, state)
    for entry in entries:
        folder = models / entry["repo"].split("/")[1]
        check(folder, entry)
        require_absent(api, entry["repo"])
        row = {"repo": entry["repo"], "status": "creating"}
        state["repositories"].append(row)
        write_receipt(receipt, state)
        api.create_repo(entry["repo"], repo_type="model", private=False, exist_ok=False)
        state["hub_writes_performed"] = True
        row["status"] = "created"
        write_receipt(receipt, state)
        info = api.model_info(entry["repo"], files_metadata=True)
        if (
            info.id != entry["repo"]
            or info.private is not False
            or not {item.rfilename for item in info.siblings} <= {".gitattributes"}
        ):
            raise ValueError("new public repository is not empty")
        commit = api.create_commit(
            entry["repo"],
            parent_commit=info.sha,
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=folder / name)
                for name in entry["files"]
            ],
            commit_message="Publish prepared classic static embeddings",
            num_threads=2,
        )
        row.update(status="uploaded", revision=commit.oid)
        write_receipt(receipt, state)
        row["readback"] = readback(entry, scratch, commit.oid)
        row["status"] = "verified"
        write_receipt(receipt, state)
    state["status"] = "complete"
    write_receipt(receipt, state)
