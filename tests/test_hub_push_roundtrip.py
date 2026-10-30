"""The chain a model goes through to be shared: converted into the fse format (a Model2Vec
or sentence-transformers model, word vectors, a classifier on such an encoder), saved as a
Hugging Face folder, pushed to the Hub and loaded back with from_pretrained, by repository
id and revision, with the same fingerprint and the same bits; and the same through a local
folder.

The Hub is a fake (``support.hub``) that commits what push_to_hub sends and serves it into
a real cache layout, so the download side runs fse's checks of commits, LFS files and cache
links unchanged. Nothing reaches the network.
"""

import os

import numpy as np
import pytest

import fse
import fse._hub as hub
from support import model2vec as mv
from support import sentence_transformers as st
from support import wordpiece as wp
from support.encoder import finished
from support.hub import FakeHub

pytest.importorskip("model2vec")
pytestmark = pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")
REPO = "acme/shared"
TEXTS = [*wp.SENTENCES.read_text(encoding="utf-8").splitlines()[:200], "", "w1 w2 zz9"]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakeHub(REPO, tmp_path / "cache")
    monkeypatch.setattr(hub, "_library", fake.library)
    return fake


def _model2vec(tmp_path):
    return fse.Encoder.from_pretrained(mv.save_model(tmp_path / "m2v"))


def _sentence_transformers(tmp_path):
    return fse.Encoder.from_pretrained(st.write_model(tmp_path / "st"), normalize=True)


def _word_vectors(_tmp_path):
    return finished(fasttext=True)


def _bits(rows):
    return rows.view(np.uint32).tobytes()


@pytest.mark.parametrize("make", [_model2vec, _sentence_transformers, _word_vectors])
def test_a_converted_encoder_is_pushed_and_loaded_back(make, fake, tmp_path):
    encoder = make(tmp_path)
    want = encoder.encode(TEXTS, workers=1)
    url = encoder.push_to_hub(REPO, license="mit")
    commit = fake.pushes[-1]
    assert url.endswith(commit)
    assert sorted(fake.commits[commit]) == [
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    for revision in (None, "main", commit):
        loaded = fse.Encoder.from_pretrained(REPO, revision=revision, cache_dir=fake.cache)
        assert loaded.fingerprint == encoder.fingerprint
        assert _bits(loaded.encode(TEXTS, workers=1)) == _bits(want)
    offline = fse.Encoder.from_pretrained(
        REPO, revision=commit, cache_dir=fake.cache, local_files_only=True
    )
    assert offline.fingerprint == encoder.fingerprint
    # The same chain through a local folder.
    folder = encoder.save_pretrained(tmp_path / "folder", license="mit")
    for mmap in (False, True):
        local = fse.Encoder.from_pretrained(folder, mmap=mmap)
        assert local.fingerprint == encoder.fingerprint
        assert _bits(local.encode(TEXTS, workers=1)) == _bits(want)


@pytest.mark.parametrize("standalone", [False, True])
def test_a_classifier_on_a_converted_encoder_is_pushed_and_loaded_back(standalone, fake, tmp_path):
    encoder = fse.Encoder.from_pretrained(mv.save_model(tmp_path / "m2v", normalize=False))
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((3, encoder.output_dim)).astype(np.float32)
    clf = fse.Classifier.from_arrays(
        encoder, weight, rng.standard_normal(3), labels=["a", "b", "c"]
    )
    labels = clf.predict(TEXTS, workers=1)
    proba = clf.predict_proba(TEXTS, workers=1)
    clf.push_to_hub(REPO, standalone=standalone, license="mit")
    commit = fake.pushes[-1]
    for revision in ("main", commit):
        loaded = fse.Classifier.from_pretrained(REPO, revision=revision, cache_dir=fake.cache)
        assert loaded.standalone is standalone
        if not standalone:
            assert loaded.fingerprint == clf.fingerprint
        assert loaded.predict(TEXTS, workers=1).tolist() == labels.tolist()
        assert _bits(loaded.predict_proba(TEXTS, workers=1)) == _bits(proba)
    folder = clf.save_pretrained(tmp_path / "folder", standalone=standalone, license="mit")
    local = fse.Classifier.from_pretrained(folder)
    assert local.fingerprint == loaded.fingerprint
    assert _bits(local.predict_proba(TEXTS, workers=1)) == _bits(proba)


def test_a_second_push_of_another_model_needs_replace_and_the_old_commit_stays(fake, tmp_path):
    first, second = _model2vec(tmp_path), finished()
    first.push_to_hub(REPO, license="mit")
    old = fake.pushes[-1]
    with pytest.raises(ValueError, match="other content under"):
        second.push_to_hub(REPO, license="mit")
    second.push_to_hub(REPO, license="mit", replace=True)
    new = fake.pushes[-1]
    assert fse.Encoder.from_pretrained(REPO, cache_dir=fake.cache).fingerprint == second.fingerprint
    by_old = fse.Encoder.from_pretrained(REPO, revision=old, cache_dir=fake.cache)
    assert by_old.fingerprint == first.fingerprint
    assert new != old
