"""A fake huggingface_hub for one repository that keeps a real cache layout (blobs named by
their sha256 or git blob sha1, snapshot files as links into them), shared by the tests of
``fse._hub``; nothing is downloaded."""

import hashlib
import os
from types import SimpleNamespace

from fse import _hub as hub
from support import model2vec as mv

COMMIT = "c" * 40
LFS = {"model.safetensors", "encoder.safetensors"}


def git_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeHub:
    """model_info and hf_hub_download of one repository, on a real cache layout."""

    def __init__(self, cache, repo, files, lfs=LFS):
        self.cache, self.repo, self.files, self.lfs = cache, repo, files, lfs
        self.pins = {}
        # Whether a download leaves refs/main. The real hf_hub_download does so only for a
        # revision by name, and fse downloads at the commit it resolved, so a cache filled by
        # fse alone has no ref; the tests keep it for the offline loads by revision name.
        self.refs = True
        self.asked = []
        self.HfApi = lambda: self  # huggingface_hub's class, which this fake stands for

    def model_info(self, repo, *, revision, files_metadata, token):
        assert files_metadata
        self.asked.append((repo, revision, token))
        siblings = []
        for name, data in self.files.items():
            lfs = (
                SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
                if name in self.lfs
                else None
            )
            blob_id, sha = self.pins.get(name, (git_sha1(data), lfs))
            siblings.append(SimpleNamespace(rfilename=name, blob_id=blob_id, lfs=sha))
        return SimpleNamespace(sha=COMMIT, siblings=siblings)

    def folder(self):
        return self.cache / ("models--" + self.repo.replace("/", "--"))

    def write_ref(self, revision):
        """``refs/<revision>``, which a download of the revision by name leaves in the cache
        (a download at a commit hash does not)."""
        ref = self.folder() / "refs" / revision
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(COMMIT)

    def hf_hub_download(self, repo, name, *, revision, local_files_only, cache_dir, token):
        assert (repo, cache_dir) == (self.repo, self.cache)
        assert token in (None, "t")
        snapshot = self.folder() / "snapshots" / COMMIT
        path = snapshot / name
        if local_files_only or os.path.lexists(path):
            if revision not in (None, "main", COMMIT) or not os.path.lexists(path):
                raise FileNotFoundError(f"{name} is not in the cache")
            return str(path)
        if self.refs:
            self.write_ref("main")
        data = self.files[name]
        blob = (
            self.folder()
            / "blobs"
            / (hashlib.sha256(data).hexdigest() if name in self.lfs else git_sha1(data))
        )
        blob.parent.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        path.symlink_to(os.path.relpath(blob, path.parent))
        return str(path)


def make_fake(tmp_path, monkeypatch):
    model = mv.save_model(tmp_path / "model")
    files = {p.name: p.read_bytes() for p in model.iterdir()}
    fake = FakeHub(tmp_path / "cache", "acme/tiny", files)
    monkeypatch.setattr(hub, "_library", lambda: fake)
    fake.model = model
    return fake
