"""StaticModel.save_pretrained and push_to_hub: the fse folder of the model's encoder, read
back by StaticModel.from_pretrained with the rows of the model before saving; the
attributes a reload rebuilds; the license; the single commit; the refusals. The Hub is a
fake (tests/support/hub.py): nothing reaches the network."""

import os
import re
import shutil

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse import _hub as hub
from fse._encoder import Encoder
from support import static_model as sm
from support.hub import FakeHub

pytest.importorskip("model2vec")
pytest.importorskip("safetensors")
REPO = "acme/static"
TEXTS = sm.texts(count=80)
LINKS = pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")


def _with_card(folder, base_model="acme/base", language=("en", "de")):
    """The README of a model2vec folder with a base_model and a language added."""
    card = folder / "README.md"
    lines = card.read_text().splitlines()
    lines[1:1] = [f"base_model: {base_model}", "language:", *(f"- {code}" for code in language)]
    card.write_text("\n".join(lines) + "\n")
    return folder


@pytest.fixture(params=sm.LAYOUTS)
def source(request, tmp_path):
    return sm.model_folder(tmp_path / "source", request.param)


def _reloaded(model, tmp_path, **options):
    folder = model.save_pretrained(tmp_path / "saved", **options)
    return folder, fse.StaticModel.from_pretrained(folder)


def _same_rows(a, b):
    assert a.encode(TEXTS).tobytes() == b.encode(TEXTS).tobytes()
    assert (
        a.encode(TEXTS, max_length=3, normalize=not a.normalize).tobytes()
        == b.encode(TEXTS, max_length=3, normalize=not a.normalize).tobytes()
    )


def test_a_saved_model_reloads_with_its_rows_and_the_rebuilt_attributes(source, tmp_path):
    model = fse.StaticModel.from_pretrained(source)
    folder, again = _reloaded(model, tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == [
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    _same_rows(model, again)
    assert again.config == {"normalize": model.normalize, "max_length": model.max_length}
    assert (again.normalize, again.max_length, again.dim) == (
        model.normalize,
        model.max_length,
        model.dim,
    )
    assert again.tokens == model.tokens
    assert again.median_token_length == model.median_token_length
    assert again.unk_token_id == model.unk_token_id
    assert again.embedding.tobytes() == model.embedding.tobytes()
    assert again.base_model_name is None  # a local source names no repository
    assert again.language is None


@pytest.mark.parametrize("flip", [True, False])
def test_setter_changes_and_a_dimensionality_cut_survive_the_reload(source, tmp_path, flip):
    model = fse.StaticModel.from_pretrained(source, dimensionality=8)
    model.normalize = not model.normalize if flip else model.normalize
    model.max_length = 7
    folder, again = _reloaded(model, tmp_path)
    assert (again.normalize, again.max_length, again.dim) == (model.normalize, 7, 8)
    _same_rows(model, again)
    model.max_length = None
    again2 = fse.StaticModel.from_pretrained(model.save_pretrained(tmp_path / "none"))
    assert again2.max_length is None
    _same_rows(model, again2)
    assert folder.name == "saved"


def test_a_model_loaded_from_an_fse_folder_saves_and_reloads(source, tmp_path):
    first = fse.StaticModel.from_pretrained(source)
    first.max_length = 11
    saved = first.save_pretrained(tmp_path / "one")
    second = fse.StaticModel.from_pretrained(saved)
    second.normalize = not second.normalize
    again = fse.StaticModel.from_pretrained(second.save_pretrained(tmp_path / "two"))
    _same_rows(second, again)
    assert (again.max_length, again.normalize) == (11, second.normalize)
    # the file of an fse encoder is a source too
    encoder_file = tmp_path / "e.safetensors"
    second.encoder.save(encoder_file)
    third = fse.StaticModel.from_pretrained(encoder_file)
    third.save_pretrained(tmp_path / "three")
    _same_rows(second, fse.StaticModel.from_pretrained(tmp_path / "three"))


def test_the_saved_folder_is_an_fse_encoder_and_maps(source, tmp_path):
    model = fse.StaticModel.from_pretrained(source)
    folder = model.save_pretrained(tmp_path / "saved", model_name="My model")
    assert "# My model" in (folder / "README.md").read_text()
    mapped = Encoder.from_pretrained(folder, mmap=True)
    assert mapped.encode(TEXTS).tobytes() == model.encode(TEXTS).tobytes()
    assert mapped.fingerprint == model.encoder.fingerprint
    default = fse.StaticModel.from_pretrained(source).save_pretrained(tmp_path / "named")
    assert "# named" in (default / "README.md").read_text()


def test_the_language_of_the_card_and_the_license_are_kept(tmp_path):
    folder = _with_card(sm.model_folder(tmp_path / "source"), language=("en", "de"))
    model = fse.StaticModel.from_pretrained(folder)
    assert model.language == ["en", "de"]
    saved, again = _reloaded(model, tmp_path)
    assert again.language == ["en", "de"]
    card = (saved / "README.md").read_text()
    assert 'license: "mit"' in card
    assert again.encoder._content.license == "mit"
    one = _with_card(sm.model_folder(tmp_path / "one"), language=("fr",))
    model = fse.StaticModel.from_pretrained(one)
    assert model.language == ["fr"]
    assert fse.StaticModel.from_pretrained(model.save_pretrained(tmp_path / "s1")).language == [
        "fr"
    ]
    plain = fse.StaticModel.from_pretrained(sm.model_folder(tmp_path / "plain"))
    assert plain.language is None


def test_a_language_fse_cannot_write_raises_instead_of_changing(tmp_path):
    folder = _with_card(sm.model_folder(tmp_path / "source"), language=("english",))
    model = fse.StaticModel.from_pretrained(folder)
    with pytest.raises(ValueError, match="no language code"):
        model.save_pretrained(tmp_path / "saved")
    assert not (tmp_path / "saved").exists()


def test_save_refuses_a_subfolder_a_present_path_and_wrong_types(source, tmp_path):
    model = fse.StaticModel.from_pretrained(source)
    with pytest.raises(ValueError, match="subfolder='sub' is not supported"):
        model.save_pretrained(tmp_path / "saved", subfolder="sub")
    assert not (tmp_path / "saved").exists()
    (tmp_path / "there").mkdir()
    with pytest.raises(FileExistsError, match=re.escape(str(tmp_path / "there"))):
        model.save_pretrained(tmp_path / "there")
    with pytest.raises(FileNotFoundError):
        model.save_pretrained(tmp_path / "missing" / "saved")
    with pytest.raises(ValidationError):
        model.save_pretrained(tmp_path / "x", model_name=3)
    with pytest.raises(ValidationError):
        model.save_pretrained(tmp_path / "x", subfolder=3)
    with pytest.raises(ValueError, match="model_name"):
        model.save_pretrained(tmp_path / "x", model_name=" ")


# --- push_to_hub ------------------------------------------------------------------------


class SpyHub(FakeHub):
    """The fake Hub that records how it was asked."""

    def __init__(self, *args):
        super().__init__(*args)
        self.created = []
        self.tokens = []

    def api(self, token=None):
        self.tokens.append(token)
        return self

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        super().create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=exist_ok)
        self.created.append(private)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = SpyHub(REPO, tmp_path / "cache")
    monkeypatch.setattr(hub, "_library", fake.library)
    return fake


def _pushed_files(fake):
    return fake.commits[fake.branches["main"]]


@LINKS
@pytest.mark.parametrize("private", [False, True])
def test_a_push_is_one_commit_of_the_fse_folder_and_reloads(fake, tmp_path, private):
    folder = _with_card(sm.model_folder(tmp_path / "source"), language=("en",))
    model = fse.StaticModel.from_pretrained(folder, dimensionality=8)
    model.normalize = False
    model.max_length = 9
    url = model.push_to_hub(REPO, private=private, token="hf_test")
    assert len(fake.pushes) == 1
    assert url == f"https://huggingface.co/{REPO}/commit/{fake.pushes[0]}"
    assert fake.created == [private]
    assert set(fake.tokens) == {"hf_test"}
    files = _pushed_files(fake)
    assert sorted(files) == [".gitattributes", "README.md", "config.json", "model.safetensors"]
    card = files["README.md"].decode()
    assert 'license: "mit"' in card
    assert '- "en"' in card
    assert f"# {REPO}" in card
    again = fse.StaticModel.from_pretrained(REPO, cache_dir=fake.cache)
    assert again.encode(TEXTS).tobytes() == model.encode(TEXTS).tobytes()
    assert (again.normalize, again.max_length, again.dim) == (False, 9, 8)
    assert again.language == ["en"]
    # the source was a local folder: its card's base_model (acme/base) is not kept
    assert model.base_model_name == "acme/base"
    assert again.base_model_name is None
    assert again.encoder._content.source.repo is None


def test_push_defaults_to_a_public_repository_without_a_token(fake, tmp_path):
    model = fse.StaticModel.from_pretrained(sm.model_folder(tmp_path / "source"))
    model.push_to_hub(REPO)
    assert fake.created == [False]
    assert fake.tokens == [None]


@LINKS
def test_the_base_model_of_a_reload_is_the_repository_the_table_came_from(fake, tmp_path):
    folder = _with_card(sm.model_folder(tmp_path / "source"), base_model="acme/original")
    fake.commits["1" * 40] = {
        ".gitattributes": b"*.safetensors filter=lfs\n",
        **{p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()},
    }
    fake.branches["main"] = "1" * 40
    model = fse.StaticModel.from_pretrained(REPO, cache_dir=fake.cache)
    assert model.base_model_name == "acme/original"
    saved = fse.StaticModel.from_pretrained(model.save_pretrained(tmp_path / "saved"))
    # fse.source names the repository the files were read from, not the base model
    assert saved.base_model_name == REPO
    assert saved.config == {"normalize": model.normalize, "max_length": model.max_length}


@pytest.mark.parametrize("layout", ["st-root", "st-nested"])
def test_a_model_without_a_license_saves_but_does_not_push(fake, tmp_path, layout):
    model = fse.StaticModel.from_pretrained(sm.model_folder(tmp_path / "source", layout))
    assert (
        "The model records no license."
        in (model.save_pretrained(tmp_path / "saved") / "README.md").read_text()
    )
    with pytest.raises(ValueError, match=r"records no license.*model\.encoder\.push_to_hub"):
        model.push_to_hub(REPO)
    assert fake.pushes == [] and fake.created == []
    model.encoder.push_to_hub(REPO, license="mit")  # the way the message names
    assert len(fake.pushes) == 1


def test_push_refuses_a_subfolder_and_a_bad_repository_before_the_hub(fake, tmp_path):
    model = fse.StaticModel.from_pretrained(sm.model_folder(tmp_path / "source"))
    with pytest.raises(ValueError, match="subfolder='sub' is not supported"):
        model.push_to_hub(REPO, subfolder="sub")
    with pytest.raises(ValueError, match="namespace/name"):
        model.push_to_hub("nonsense")
    with pytest.raises(ValidationError):
        model.push_to_hub(REPO, private="yes")
    with pytest.raises(ValidationError):
        model.push_to_hub(3)
    assert fake.pushes == [] and fake.created == []


def test_a_push_never_replaces_other_content(fake, tmp_path):
    model = fse.StaticModel.from_pretrained(sm.model_folder(tmp_path / "source"))
    model.push_to_hub(REPO)
    model.normalize = not model.normalize
    with pytest.raises(ValueError, match="replace=True"):
        model.push_to_hub(REPO)
    assert len(fake.pushes) == 1


def test_the_folder_is_a_copy_of_the_model_not_a_view(tmp_path):
    source = sm.model_folder(tmp_path / "source")
    model = fse.StaticModel.from_pretrained(source)
    before = model.encode(TEXTS).tobytes()
    saved = model.save_pretrained(tmp_path / "saved")
    shutil.rmtree(source)
    assert model.encode(TEXTS).tobytes() == before
    assert np.array_equal(fse.StaticModel.from_pretrained(saved).encode(TEXTS), model.encode(TEXTS))
