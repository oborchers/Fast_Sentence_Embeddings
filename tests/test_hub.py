"""Hub repositories: one commit, every file checked against it, cache links under control.

huggingface_hub is replaced by a fake that keeps a real cache layout (blobs named by
their sha256 or git blob sha1, snapshot files as links into them), so nothing is
downloaded; the network tests use the real Hub.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import fse._hub as hub
from fse._container import CorruptArtifactError
from fse._encoder import Encoder
from fse._encoder._format import Source
from support import model2vec as mv

pytest.importorskip("model2vec")
pytestmark = pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")
COMMIT = "c" * 40
LFS = {"model.safetensors", "encoder.safetensors"}


def _sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeHub:
    """model_info and hf_hub_download of one repository, on a real cache layout."""

    def __init__(self, cache, repo, files, lfs=LFS):
        self.cache, self.repo, self.files, self.lfs = cache, repo, files, lfs
        self.pins = {}
        self.asked = []
        self.HfApi = lambda: self  # huggingface_hub's class, which this fake stands for

    def model_info(self, repo, *, revision, files_metadata, token):
        assert files_metadata
        self.asked.append((repo, revision, token))
        siblings = []
        for name, data in self.files.items():
            lfs = (
                SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
                if name in self.lfs
                else None
            )
            blob_id, sha = self.pins.get(name, (_sha1(data), lfs))
            siblings.append(SimpleNamespace(rfilename=name, blob_id=blob_id, lfs=sha))
        return SimpleNamespace(sha=COMMIT, siblings=siblings)

    def folder(self):
        return self.cache / ("models--" + self.repo.replace("/", "--"))

    def hf_hub_download(self, repo, name, *, revision, local_files_only, cache_dir, token):
        assert (repo, cache_dir) == (self.repo, self.cache)
        assert token in (None, "t")
        snapshot = self.folder() / "snapshots" / COMMIT
        path = snapshot / name
        if local_files_only or os.path.lexists(path):
            if revision not in (None, "main", COMMIT) or not os.path.lexists(path):
                raise FileNotFoundError(f"{name} is not in the cache")
            return str(path)
        data = self.files[name]
        blob = (
            self.folder()
            / "blobs"
            / (hashlib.sha256(data).hexdigest() if name in self.lfs else _sha1(data))
        )
        blob.parent.mkdir(parents=True, exist_ok=True)
        snapshot.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        path.symlink_to(os.path.relpath(blob, snapshot))
        return str(path)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    model = mv.save_model(tmp_path / "model")
    files = {p.name: p.read_bytes() for p in model.iterdir()}
    fake = FakeHub(tmp_path / "cache", "acme/tiny", files)
    monkeypatch.setattr(hub, "_library", lambda: fake)
    fake.model = model
    return fake


def _load(fake, **options):
    return Encoder.from_pretrained("acme/tiny", cache_dir=fake.cache, **options)


def test_a_model_of_the_hub_is_converted_at_its_commit(fake):
    encoder = _load(fake, revision="main", token="t")
    assert fake.asked == [("acme/tiny", "main", "t")]
    assert encoder._content.source == Source("model2vec", "acme/tiny", COMMIT)
    assert encoder._content.license == "mit"
    texts = ["hello world", "ab"]
    assert (
        encoder.encode(texts).tobytes()
        == Encoder.from_pretrained(fake.model).encode(texts).tobytes()
    )
    assert _load(fake).fingerprint == encoder.fingerprint  # the same commit, the same encoder


def test_a_changed_blob_is_caught(fake):
    _load(fake)
    blob = os.path.realpath(fake.folder() / "snapshots" / COMMIT / "model.safetensors")
    with open(blob, "r+b") as f:
        f.seek(-4, 2)
        f.write(b"\1\2\3\4")
    with pytest.raises(
        CorruptArtifactError, match=r"model\.safetensors has sha256 .* the commit pins"
    ):
        _load(fake)
    with pytest.raises(CorruptArtifactError, match=r"model\.safetensors: the blob .* has sha256"):
        _load(fake, local_files_only=True)


def test_a_safetensors_file_must_be_an_lfs_file(fake):
    fake.lfs = set()
    with pytest.raises(CorruptArtifactError, match=r"no LFS file with an lfs\.sha256"):
        _load(fake)


def test_a_small_file_is_checked_by_its_git_blob_id(fake):
    fake.pins["config.json"] = ("0" * 40, None)
    with pytest.raises(
        CorruptArtifactError, match=r"config\.json has the git blob id .* pins 0000"
    ):
        _load(fake)
    fake.pins["config.json"] = (None, None)
    with pytest.raises(CorruptArtifactError, match="pins None"):
        _load(fake)


def test_local_files_only_reads_the_cache(fake):
    online = _load(fake)
    fake.asked.clear()
    offline = _load(fake, local_files_only=True)
    assert fake.asked == []
    assert offline.fingerprint == online.fingerprint
    snapshot = fake.folder() / "snapshots" / COMMIT
    blob = os.path.realpath(snapshot / "config.json")
    (snapshot / "config.json").unlink()
    shutil.copy(blob, snapshot / "config.json")
    with pytest.raises(CorruptArtifactError, match="a copy without its blob name"):
        _load(fake, local_files_only=True)
    _load(fake)  # online, the copy is checked against the commit


def test_a_link_out_of_the_blobs_is_refused(fake, tmp_path):
    _load(fake)
    snapshot = fake.folder() / "snapshots" / COMMIT
    outside = tmp_path / "config.json"
    outside.write_bytes(fake.files["config.json"])
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(outside)
    with pytest.raises(CorruptArtifactError, match="outside the blobs"):
        _load(fake)


def test_the_name_of_a_cached_blob_is_checked(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    sha256, sha1 = hashlib.sha256(b"x").hexdigest(), _sha1(b"x")
    hub._check_name("f", sha256, sha256, sha1, True)
    hub._check_name("f", sha1, sha256, sha1, False)
    with pytest.raises(CorruptArtifactError, match="no LFS file of the cache"):
        hub._check_name("f", sha1, sha256, sha1, True)
    with pytest.raises(CorruptArtifactError, match="has the git blob id"):
        hub._check_name("f", "1" * 40, sha256, sha1, False)


def test_an_fse_encoder_on_the_hub_is_an_lfs_file(fake, tmp_path):
    encoder = Encoder.from_pretrained(fake.model)
    encoder.save(tmp_path / "encoder.safetensors")
    config = {
        "model_type": "fse",
        "fse_kind": "encoder",
        "fse_format_version": 1,
        "encoder": "encoder.safetensors",
    }
    fake.files = {
        "config.json": json.dumps(config).encode(),
        "encoder.safetensors": (tmp_path / "encoder.safetensors").read_bytes(),
    }
    assert _load(fake).fingerprint == encoder.fingerprint
    with pytest.raises(ValueError, match="mmap is for local fse encoders"):
        _load(fake, mmap=True)
    fake.cache = tmp_path / "other-cache"
    fake.lfs = set()
    with pytest.raises(
        CorruptArtifactError, match=r"encoder\.safetensors of commit .* no LFS file"
    ):
        _load(fake)


def test_repository_ids_and_the_extra(monkeypatch):
    with pytest.raises(FileNotFoundError, match="nor a Hub repository id"):
        Encoder.from_pretrained("no/such/thing")
    with pytest.raises(ValueError, match="no Hub repository id"):
        hub.HubFiles("a/b/c", revision=None, token=None, cache_dir=None, local_files_only=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r'pip install "fse\[hub\]"'):
        Encoder.from_pretrained("acme/tiny")


def _share(fake, name, store_name=None):
    """Move a blob into the cache-wide store of huggingface-hub 2 and link it, as the
    Xet files of that version are kept."""
    snapshot = fake.folder() / "snapshots" / COMMIT
    blob = Path(os.path.normpath(snapshot / os.readlink(snapshot / name)))
    store_name = store_name or hashlib.sha256(b"xet" + blob.name.encode()).hexdigest()
    shared = fake.cache / "blobs" / store_name[:2] / store_name
    shared.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(blob, shared)
    blob.symlink_to(os.path.relpath(shared, blob.parent))
    return shared


def test_the_shared_blobs_of_huggingface_hub_2_are_followed(fake):
    online = _load(fake)
    _share(fake, "model.safetensors")
    assert _load(fake).fingerprint == online.fingerprint
    assert _load(fake, local_files_only=True).fingerprint == online.fingerprint


def test_a_shared_blob_outside_the_store_is_refused(fake, tmp_path):
    _load(fake)
    snapshot = fake.folder() / "snapshots" / COMMIT
    blob = Path(os.path.normpath(snapshot / os.readlink(snapshot / "model.safetensors")))
    elsewhere = tmp_path / "elsewhere"
    shutil.move(blob, elsewhere)
    blob.symlink_to(elsewhere)
    with pytest.raises(CorruptArtifactError, match="outside the shared blobs"):
        _load(fake)
    blob.unlink()
    shutil.move(elsewhere, blob)
    _share(fake, "model.safetensors", store_name="x" * 64)
    with pytest.raises(CorruptArtifactError, match="outside the shared blobs"):
        _load(fake)


def test_a_blob_replaced_after_its_check_changes_nothing(fake, monkeypatch):
    """The bytes that are checked are the bytes that are parsed: a blob replaced right
    after its check (between check and load, if they read the file twice) is not read."""
    online = _load(fake)
    blob = os.path.realpath(fake.folder() / "snapshots" / COMMIT / "model.safetensors")
    original, swapped = hub._check_pin, []

    def check_then_swap(name, *args):
        original(name, *args)
        if name == "model.safetensors":
            data = bytearray(Path(blob).read_bytes())
            data[-4:] = b"\1\2\3\4"
            Path(blob).write_bytes(bytes(data))
            swapped.append(name)

    monkeypatch.setattr(hub, "_check_pin", check_then_swap)
    again = _load(fake)
    assert swapped == ["model.safetensors"]
    assert again.encode(["hello world"]).tobytes() == online.encode(["hello world"]).tobytes()
    monkeypatch.setattr(hub, "_check_pin", original)
    with pytest.raises(CorruptArtifactError, match="the commit pins"):
        _load(fake)
