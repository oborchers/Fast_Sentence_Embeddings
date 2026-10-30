"""A fake Hugging Face Hub of one repository, for both directions: push_to_hub commits
into it (create_repo, model_info, create_commit) and from_pretrained downloads from it
into a real cache layout (blobs named by their sha256 or git blob sha1, snapshot files as
links into them). Nothing reaches the network. The commit operations are huggingface_hub's
own classes, so their checks run."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub

EMPTY = "0" * 40


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeHub:
    """The Hub of ``repo`` with its commits; ``library()`` stands for huggingface_hub."""

    def __init__(self, repo: str, cache: Path):
        self.repo, self.cache = repo, Path(cache)
        self.commits = {EMPTY: {".gitattributes": b"*.safetensors filter=lfs\n"}}
        self.branches = {"main": EMPTY}
        self.pushes = []

    def library(self):
        return SimpleNamespace(
            HfApi=self.api,
            CommitOperationAdd=huggingface_hub.CommitOperationAdd,
            CommitOperationDelete=huggingface_hub.CommitOperationDelete,
            constants=SimpleNamespace(ENDPOINT="https://huggingface.co"),
            hf_hub_download=self.hf_hub_download,
            snapshot_download=self.snapshot_download,
        )

    def api(self, token=None):  # huggingface_hub.HfApi(token=...)
        del token
        return self

    # --- what push_to_hub uses ----------------------------------------------------

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        assert (repo_id, repo_type, exist_ok) == (self.repo, "model", True)
        del private

    def _commit_of(self, revision):
        if revision in self.commits:
            return revision
        return self.branches[revision or "main"]

    def model_info(self, repo_id, *, revision=None, files_metadata=False, token=None):
        assert repo_id == self.repo
        assert files_metadata
        del token
        commit = self._commit_of(revision)
        return SimpleNamespace(
            sha=commit, siblings=[_sibling(n, d) for n, d in self.commits[commit].items()]
        )

    def create_commit(self, repo_id, operations, **options):
        assert repo_id == self.repo
        assert not options["create_pr"]  # pull requests: the fake of tests/test_hub_push.py
        branch = options["revision"] or "main"
        if options["parent_commit"] != self.branches[branch]:
            raise RuntimeError("the branch moved since the parent commit")
        files = dict(self.commits[self.branches[branch]])
        for op in operations:
            if isinstance(op, huggingface_hub.CommitOperationDelete):
                del files[op.path_in_repo]
            else:
                files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
        oid = hashlib.sha1(repr(sorted(files.items())).encode()).hexdigest()
        self.commits[oid] = files
        self.branches[branch] = oid
        self.pushes.append(oid)
        return SimpleNamespace(
            oid=oid, commit_url=f"https://huggingface.co/{repo_id}/commit/{oid}", pr_url=None
        )

    # --- what from_pretrained uses ------------------------------------------------

    def folder(self, cache=None) -> Path:
        return Path(cache or self.cache) / ("models--" + self.repo.replace("/", "--"))

    def hf_hub_download(
        self, repo, name, *, revision, cache_dir, token, local_files_only=False
    ):  # the loader passes local_files_only, push_to_hub reads .gitattributes without it
        assert repo == self.repo
        del token
        commit = self._commit_of(revision)
        folder = self.folder(cache_dir)
        path = folder / "snapshots" / commit / name
        if os.path.lexists(path):
            return str(path)
        if local_files_only:
            raise FileNotFoundError(f"{name} is not in the cache")
        data = self.commits[commit][name]
        etag = hashlib.sha256(data).hexdigest() if name.endswith(".safetensors") else git_sha1(data)
        blob = folder / "blobs" / etag
        blob.parent.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        path.symlink_to(os.path.relpath(blob, path.parent))
        return str(path)

    def snapshot_download(self, repo, *, revision, local_files_only, cache_dir, token):
        assert local_files_only
        del repo, cache_dir, token
        snapshot = self.folder() / "snapshots" / self._commit_of(revision)
        if not snapshot.is_dir():
            raise FileNotFoundError(f"{self.repo} is not in the cache")
        return str(snapshot)


def _sibling(name: str, data: bytes) -> SimpleNamespace:
    """A file as the Hub stores what a push adds: .gitattributes makes every
    ``.safetensors`` file an LFS file (the fake of tests/test_hub_push.py also models
    files stored otherwise)."""
    if name.endswith(".safetensors"):
        lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        return SimpleNamespace(rfilename=name, lfs=lfs, blob_id=git_sha1(b"pointer " + data))
    return SimpleNamespace(rfilename=name, lfs=None, blob_id=git_sha1(data))
