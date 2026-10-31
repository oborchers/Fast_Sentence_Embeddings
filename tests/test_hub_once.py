"""A file of a Hub commit is hashed once, with the one algorithm its pin names.

The pin of a file of the commit is its lfs.sha256 or its git blob sha1 (offline: its blob
name); the other digest cannot match anything, so it is not computed. The constructors of
hashlib are watched, not only the calls of ``pin_digest``, because a call could compute
both.
"""

import os
from types import SimpleNamespace

import pytest

import fse._hub as hub
from fse._container import CorruptArtifactError, _hash
from fse._encoder import Encoder
from fse._encoder._format import Source
from support.hubcache import COMMIT, LFS, make_fake
from support.logs import captured

pytest.importorskip("model2vec")
pytestmark = pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")
SHA256, GIT = "lfs_sha256", "git_sha1"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return make_fake(tmp_path, monkeypatch)


@pytest.fixture
def constructed(monkeypatch):
    """The hashlib constructors fse calls, in order."""
    calls = []

    def spy(name):
        def construct(*args):
            calls.append(name)
            return getattr(__import__("hashlib"), name)(*args)

        return construct

    monkeypatch.setattr(_hash, "hashlib", SimpleNamespace(sha256=spy("sha256"), sha1=spy("sha1")))
    return calls


def _load(fake, **options):
    return Encoder.from_pretrained("acme/tiny", cache_dir=fake.cache, **options)


def _files(fake, offline=False):
    return hub.HubFiles(
        "acme/tiny", revision=None, token=None, cache_dir=fake.cache, local_files_only=offline
    )


def _pins_read(load):
    """The ``pin`` of every file the load took, in order."""
    with captured() as records:
        load()
    return [r.fse_pin for r in records if r.fse_event == "hub.file"]


@pytest.mark.parametrize("lfs", [LFS, LFS | {"tokenizer.json"}], ids=["tokenizer in git", "LFS"])
@pytest.mark.parametrize("offline", [False, True])
def test_each_file_is_hashed_once_with_the_algorithm_of_its_pin(fake, constructed, lfs, offline):
    fake.lfs = lfs
    if offline:
        _load(fake)  # fills the cache
        constructed.clear()
    pins = _pins_read(lambda: _load(fake, local_files_only=offline))
    assert pins and set(pins) == {SHA256, GIT}
    assert constructed.count("sha256") == pins.count(SHA256)
    assert constructed.count("sha1") == pins.count(GIT)
    assert len(constructed) == len(pins)  # one constructor per file
    assert pins.count(SHA256) == len(LFS & set(fake.files)) + ("tokenizer.json" in lfs)


def test_an_lfs_file_that_is_no_safetensors_file_is_logged_by_its_sha256(fake):
    fake.lfs = LFS | {"tokenizer.json"}
    _load(fake)
    for offline in (False, True):
        files = _files(fake, offline)
        with captured() as records:
            files.read("tokenizer.json")
        assert [r.fse_pin for r in records if r.fse_event == "hub.file"] == [SHA256]


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json"])
@pytest.mark.parametrize("lfs_tokenizer", [False, True])
def test_a_changed_cached_file_is_caught_online_and_offline(fake, lfs_tokenizer, name):
    if lfs_tokenizer:
        fake.lfs = LFS | {"tokenizer.json"}
    _load(fake)
    blob = os.path.realpath(fake.folder() / "snapshots" / COMMIT / name)
    with open(blob, "r+b") as f:  # the size stays, which the commit records for LFS files
        f.seek(-1, 2)
        f.write(b"\1")
    lfs = lfs_tokenizer and name == "tokenizer.json"
    online = "has sha256 .* the commit pins" if lfs else "has the git blob id .* pins"
    offline = "the blob .* has sha256" if lfs else "the blob .* has the git blob id"
    with pytest.raises(CorruptArtifactError, match=online):
        _load(fake)
    with pytest.raises(CorruptArtifactError, match=offline):
        _load(fake, local_files_only=True)


def test_a_file_is_refused_before_it_is_hashed(fake, constructed):
    _load(fake)
    constructed.clear()
    with pytest.raises(CorruptArtifactError, match="no LFS file of the cache"):
        _files(fake, offline=True).read("config.json", lfs=True)  # a git file by its name
    with pytest.raises(CorruptArtifactError, match="no LFS file with an lfs.sha256"):
        _files(fake).read("config.json", lfs=True)
    assert constructed == []
    fake.pins["config.json"] = (None, None)  # no blob id: the message names the digest
    with pytest.raises(CorruptArtifactError, match="pins None"):
        _files(fake).read("config.json")
    assert constructed == ["sha1"]


def test_a_local_folder_is_not_hashed(fake, constructed, tmp_path):
    assert Encoder.from_pretrained(fake.model).fingerprint
    snapshot = fake.folder() / "snapshots"
    _load(fake)
    constructed.clear()
    assert Encoder.from_pretrained(snapshot / COMMIT).fingerprint
    assert Encoder.from_pretrained(fake.model).fingerprint
    assert constructed == []


@pytest.mark.parametrize("offline", [False, True])
def test_an_explicit_commit_is_read_and_the_pin_of_other_bytes_is_refused(fake, offline):
    _load(fake)
    encoder = _load(fake, revision=COMMIT, local_files_only=offline)
    assert encoder._content.source == Source("model2vec", "acme/tiny", COMMIT)
    asked = len(fake.asked)
    _load(fake, revision="main", local_files_only=offline)
    assert len(fake.asked) == asked + (not offline)  # a branch is resolved once
    if not offline:
        fake.pins["model.safetensors"] = (
            "0" * 40,
            SimpleNamespace(sha256="0" * 64, size=len(fake.files["model.safetensors"])),
        )
        with pytest.raises(CorruptArtifactError, match="the commit pins"):
            _load(fake, revision=COMMIT)
