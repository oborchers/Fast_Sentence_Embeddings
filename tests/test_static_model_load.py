"""Where StaticModel.from_pretrained finds a model: folders, Hub repositories (a fake
Hub, no network), subfolders, and fse encoders, which load if they fit the profile."""

import os

import numpy as np
import pytest

import fse
from fse import _hub as hub
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from support import static_model as sm
from support.hub import FakeHub

model2vec = pytest.importorskip("model2vec")
pytest.importorskip("safetensors")
REPO = "acme/static"
TEXTS = sm.texts(count=80)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakeHub(REPO, tmp_path / "cache")
    monkeypatch.setattr(hub, "_library", fake.library)
    return fake


def _commit(fake, folder, prefix=""):
    """The files of ``folder`` as a commit of the fake repository (under ``prefix``)."""
    files = {
        prefix + path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }
    commit = f"{len(fake.commits):040x}"
    fake.commits[commit] = {".gitattributes": b"*.safetensors filter=lfs\n", **files}
    fake.branches["main"] = commit
    return commit


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")
@pytest.mark.parametrize("layout", sm.LAYOUTS)
def test_a_hub_repository_gives_the_rows_of_a_folder(fake, tmp_path, layout):
    folder = sm.model_folder(tmp_path / "m", layout)
    commit = _commit(fake, folder)
    want = fse.StaticModel.from_pretrained(folder).encode(TEXTS)
    for revision in (None, "main", commit):
        model = fse.StaticModel.from_pretrained(REPO, revision=revision, cache_dir=fake.cache)
        assert model.encode(TEXTS).tobytes() == want.tobytes()
    offline = fse.StaticModel.from_pretrained(
        REPO, revision=commit, cache_dir=fake.cache, local_files_only=True
    )
    assert offline.encode(TEXTS).tobytes() == want.tobytes()
    source = model.encoder._content.source
    assert (source.format, source.repo, source.revision) == ("model2vec", REPO, commit)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")
def test_a_subfolder_of_a_hub_repository(fake, tmp_path):
    folder = sm.model_folder(tmp_path / "m", "st-nested")
    _commit(fake, folder, prefix="sub/dir/")
    want = fse.StaticModel.from_pretrained(folder).encode(TEXTS)
    model = fse.StaticModel.from_pretrained(REPO, subfolder="sub/dir", cache_dir=fake.cache)
    assert model.encode(TEXTS).tobytes() == want.tobytes()
    with pytest.raises(UnsupportedArtifactError, match="none of the folder layouts"):
        fse.StaticModel.from_pretrained(REPO, cache_dir=fake.cache)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")
def test_an_fse_folder_of_a_hub_repository_loads(fake, tmp_path):
    encoder = Encoder.from_pretrained(sm.model_folder(tmp_path / "m"))
    _commit(fake, encoder.save_pretrained(tmp_path / "fse", license="mit"))
    model = fse.StaticModel.from_pretrained(REPO, cache_dir=fake.cache)
    assert model.encoder.fingerprint == encoder.fingerprint
    assert model.encode(TEXTS).tobytes() == encoder.encode(TEXTS).tobytes()
    with pytest.raises(UnsupportedArtifactError, match="none of the folder layouts"):
        fse.StaticModel.from_pretrained(REPO, subfolder="a", cache_dir=fake.cache)


@pytest.mark.parametrize("how", ["file", "folder"])
def test_an_fse_encoder_loads_with_the_settings_it_has(tmp_path, how):
    encoder = Encoder.from_pretrained(sm.model_folder(tmp_path / "m"), max_length=9)
    if how == "file":
        encoder.save(tmp_path / "e.safetensors")
        path = tmp_path / "e.safetensors"
    else:
        path = encoder.save_pretrained(tmp_path / "fse", license="mit")
    model = fse.StaticModel.from_pretrained(path)
    assert (model.max_length, model.normalize) == (9, True)
    assert model.encoder.fingerprint == encoder.fingerprint
    assert model.encode(TEXTS).tobytes() == encoder.encode(TEXTS).tobytes()
    assert model.config == {"max_length": 9, "normalize": True}
    assert (model.base_model_name, model.language) == (None, None)
    assert tuple(model.tokens) == tuple(
        encoder.word_vectors.word(i) for i in range(len(encoder.word_vectors))
    )


def test_the_settings_of_an_fse_encoder_can_be_overridden_but_not_its_folder(tmp_path):
    encoder = Encoder.from_pretrained(sm.model_folder(tmp_path / "m"), max_length=9)
    encoder.save(tmp_path / "e.safetensors")
    path = tmp_path / "e.safetensors"
    model = fse.StaticModel.from_pretrained(path, normalize=False, max_length=None)
    assert (model.max_length, model.normalize) == (None, False)
    want = Encoder.from_pretrained(tmp_path / "m", truncate=False).encode(TEXTS)
    rows = model.encode(TEXTS, normalize=True)
    assert rows.shape == want.shape
    assert np.allclose(rows, want, atol=1e-5)
    narrow = fse.StaticModel.from_pretrained(path, dimensionality=5)
    assert narrow.dim == 5
    assert np.array_equal(narrow.embedding, encoder.word_vectors.stored_vectors[:, :5])
    with pytest.raises(ValueError, match="not for an fse model"):
        fse.StaticModel.from_pretrained(path, subfolder="a")
    with pytest.raises(ValueError, match="are for the Hub"):
        fse.StaticModel.from_pretrained(path, local_files_only=True)


def test_force_download_changes_nothing(tmp_path):
    folder = sm.model_folder(tmp_path / "m")
    a = fse.StaticModel.from_pretrained(folder, force_download=True)
    b = fse.StaticModel.from_pretrained(folder, force_download=False)
    assert a.encode(TEXTS).tobytes() == b.encode(TEXTS).tobytes()


def test_a_second_cut_of_an_fse_encoder_records_the_width_of_the_model_it_came_from(tmp_path):
    folder = sm.model_folder(tmp_path / "m")
    cut = Encoder.from_pretrained(folder)
    first = fse.StaticModel.from_pretrained(folder, dimensionality=12)
    assert first.encoder._content.source.dims == (16, 12)
    first.encoder.save(tmp_path / "a.safetensors")
    second = fse.StaticModel.from_pretrained(tmp_path / "a.safetensors", dimensionality=6)
    assert second.encoder._content.source.dims == (16, 6)
    assert second.dim == 6
    assert cut.output_dim == 16
