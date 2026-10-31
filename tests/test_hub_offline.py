"""``local_files_only``: the cached snapshot of a revision is found in the layout of the
cache, and only the files the model reads must be there.

DR-183 (Laden mit local_files_only bei unvollständigem Snapshot klären):
``snapshot_download(local_files_only=True)`` of huggingface_hub 2 raises when any file of
the cached tree listing of the commit is missing, although fse needs none of them.
"""

import json
import os

import pytest

import fse._hub as hub
from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from support.hubcache import COMMIT, make_fake

pytest.importorskip("model2vec")
pytestmark = pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return make_fake(tmp_path, monkeypatch)


def _load(fake, **options):
    return Encoder.from_pretrained("acme/tiny", cache_dir=fake.cache, **options)


def _snapshot(fake):
    return fake.folder() / "snapshots" / COMMIT


def _files(fake, revision=None):
    return hub.HubFiles(
        "acme/tiny", revision=revision, token=None, cache_dir=fake.cache, local_files_only=True
    )


def test_files_of_the_commit_that_were_never_downloaded_do_not_matter(fake):
    """The cache of the DR-144 benchmark: the commit lists evaluation files (25 of
    minishlab/potion-multilingual-128M) that fse never reads."""
    fake.files = {**fake.files, **{f".eval_results/{i}.json": b"{}" for i in range(3)}}
    online = _load(fake)
    assert not (_snapshot(fake) / ".eval_results").exists()
    offline = _load(fake, local_files_only=True)
    assert offline.fingerprint == online.fingerprint
    files = _files(fake)
    assert not any(name.startswith(".eval_results") for name in files._names)
    assert files.read("config.json")


def test_a_file_the_model_needs_that_is_missing_raises(fake):
    """The snapshot lacks a file of the Model2Vec layout: it is not read as one, and the
    load raises instead of using what is left."""
    del fake.files["modules.json"]  # a Model2Vec repository without a sentence-transformers one
    online = _load(fake)
    (_snapshot(fake) / "model.safetensors").unlink()
    with pytest.raises(UnsupportedArtifactError, match="no static embedding model fse reads"):
        _load(fake, local_files_only=True)
    assert not _files(fake).has("model.safetensors")
    with pytest.raises(CorruptArtifactError, match="not a file of the commit"):
        _files(fake).read("model.safetensors")
    _load(fake)  # online, the file comes back
    assert _load(fake, local_files_only=True).fingerprint == online.fingerprint


def test_a_link_of_the_snapshot_without_its_blob_raises(fake):
    _load(fake)
    os.unlink(os.path.realpath(_snapshot(fake) / "model.safetensors"))
    with pytest.raises((CorruptArtifactError, FileNotFoundError), match="blobs"):
        _load(fake, local_files_only=True)


@pytest.mark.parametrize("revision", [None, "main", COMMIT])
def test_a_revision_name_and_a_commit_hash_load_offline(fake, revision):
    online = _load(fake)
    assert _files(fake, revision).revision == COMMIT
    assert _load(fake, revision=revision, local_files_only=True).fingerprint == online.fingerprint


def test_a_branch_with_a_slash_and_a_tag_are_refs(fake):
    _load(fake)
    fake.write_ref("refs/pr/1")
    fake.write_ref("v1")
    assert _files(fake, "refs/pr/1").revision == COMMIT
    assert _files(fake, "v1").revision == COMMIT


def test_a_ref_of_another_commit_gives_its_snapshot(fake):
    _load(fake)
    other = fake.folder() / "snapshots" / ("d" * 40)
    os.rename(_snapshot(fake), other)
    (fake.folder() / "refs" / "main").write_text("d" * 40)
    assert _files(fake).revision == "d" * 40


@pytest.mark.parametrize("revision", ["dev", "e" * 40, "refs/pr/9"])
def test_no_ref_or_snapshot_raises_file_not_found(fake, revision):
    _load(fake)
    with pytest.raises(FileNotFoundError, match="local_files_only"):
        _files(fake, revision)


def test_a_cache_filled_by_fse_alone_has_no_ref_and_loads_by_commit(fake):
    """fse downloads at the commit it resolved, which leaves no refs/main (as before this
    change): a revision name needs a ref, the commit hash does not."""
    fake.refs = False
    online = _load(fake)
    assert not (fake.folder() / "refs").exists()
    with pytest.raises(FileNotFoundError, match="no ref 'main'"):
        _load(fake, local_files_only=True)
    offline = _load(fake, revision=COMMIT, local_files_only=True)
    assert offline.fingerprint == online.fingerprint


def test_an_empty_cache_raises_file_not_found(fake):
    with pytest.raises(FileNotFoundError, match="no ref 'main'"):
        _files(fake)


def test_a_ref_without_its_snapshot_raises_file_not_found(fake):
    fake.write_ref("main")
    with pytest.raises(FileNotFoundError, match="no snapshot of commit"):
        _files(fake)


@pytest.mark.parametrize("revision", ["", "..", "../x", "a/../b", "a//b", "a\\b", "./a"])
def test_a_revision_that_leads_out_of_refs_raises(fake, revision):
    _load(fake)
    with pytest.raises(ValueError, match="no revision name"):
        _files(fake, revision)


def test_a_ref_that_is_no_commit_hash_raises(fake):
    _load(fake)
    (fake.folder() / "refs" / "main").write_text("../../elsewhere")
    with pytest.raises(CorruptArtifactError, match="no commit hash"):
        _files(fake)


def test_a_ref_that_is_a_link_raises(fake, tmp_path):
    _load(fake)
    ref = fake.folder() / "refs" / "main"
    elsewhere = tmp_path / "ref"
    elsewhere.write_text(COMMIT)
    ref.unlink()
    ref.symlink_to(elsewhere)
    with pytest.raises(CorruptArtifactError, match="no link in the cache"):
        _files(fake)


def test_a_folder_of_the_ref_that_is_a_link_raises(fake, tmp_path):
    _load(fake)
    fake.write_ref("refs/pr/1")
    elsewhere = tmp_path / "elsewhere"
    os.rename(fake.folder() / "refs" / "refs", elsewhere)
    (fake.folder() / "refs" / "refs").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(CorruptArtifactError, match="is a link"):
        _files(fake, "refs/pr/1")
    refs = fake.folder() / "refs"
    os.rename(refs, tmp_path / "all-refs")
    refs.symlink_to(tmp_path / "all-refs", target_is_directory=True)
    with pytest.raises(CorruptArtifactError, match="is a link"):
        _files(fake)


def test_a_snapshot_that_is_a_link_raises(fake, tmp_path):
    _load(fake)
    elsewhere = tmp_path / "elsewhere"
    os.rename(_snapshot(fake), elsewhere)
    _snapshot(fake).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(CorruptArtifactError, match="is a link"):
        _files(fake)


def test_the_default_cache_is_that_of_huggingface_hub(fake):
    _load(fake)
    library = type("Library", (), {"constants": type("C", (), {"HF_HUB_CACHE": str(fake.cache)})})
    assert hub.cached_snapshot(library, "acme/tiny", None, None) == _snapshot(fake)


def test_the_real_huggingface_hub_refuses_the_snapshot_that_fse_loads(fake, monkeypatch):
    """The cause of DR-183, with the installed huggingface_hub: its cached tree listing
    names a file that is not in the snapshot. snapshot_download (huggingface_hub 2) raises
    for it; fse does not use it, and the real hf_hub_download serves the needed files."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    tree = fake.folder() / "trees" / f"{COMMIT}.json"
    entry = {"size": 2, "blob_id": "a" * 40}
    online = _load(fake)
    listing = {name: {"size": 1, "blob_id": "b" * 40} for name in fake.files}
    listing[".eval_results/0.json"] = entry
    tree.parent.mkdir(parents=True, exist_ok=True)
    tree.write_text(json.dumps({"format_version": 1, "files": listing}))
    monkeypatch.setattr(hub, "_library", lambda: huggingface_hub)
    offline = _load(fake, local_files_only=True)
    assert offline.fingerprint == online.fingerprint
    try:
        folder = huggingface_hub.snapshot_download(
            "acme/tiny", local_files_only=True, cache_dir=fake.cache
        )
    except FileNotFoundError:  # IncompleteSnapshotError, huggingface_hub 2 and newer
        return
    assert folder == str(_snapshot(fake))  # huggingface_hub 1 checks no listing


def test_a_cache_folder_with_a_tilde_is_expanded(fake, tmp_path, monkeypatch):
    _load(fake)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    home.mkdir()
    os.rename(fake.cache, home / "hf-cache")
    files = hub.HubFiles(
        "acme/tiny", revision=COMMIT, token=None, cache_dir="~/hf-cache", local_files_only=True
    )
    assert files.revision == COMMIT
