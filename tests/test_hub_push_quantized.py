"""A quantized encoder goes the way of every model: saved and loaded (mapped or not),
written as a Hugging Face folder with its card, pushed to the (fake) Hub and loaded back
from there, for every kind of tokenizer and both dtypes, with the same fingerprint and
the same bits."""

import os

import numpy as np
import pytest

import fse
import fse._hub as hub
from support import model2vec as mv
from support import sentence_transformers as st
from support import unigram as uni
from support import wordpiece as wp
from support.encoder import finished
from support.hub import FakeHub

pytest.importorskip("model2vec")
pytestmark = pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")
REPO = "acme/quantized"
TEXTS = [*wp.SENTENCES.read_text(encoding="utf-8").splitlines()[:150], "", "w1 w2 zz9"]
UNIGRAM_TEXTS = uni.corpus(150, seed=2)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakeHub(REPO, tmp_path / "cache")
    monkeypatch.setattr(hub, "_library", fake.library)
    return fake


def _wordpiece(tmp_path):
    return fse.Encoder.from_pretrained(mv.save_model(tmp_path / "m2v", max_length=16)), TEXTS


def _unigram(tmp_path):
    pytest.importorskip("tokenizers")
    return fse.Encoder.from_pretrained(mv.save_unigram_model(tmp_path / "uni")), UNIGRAM_TEXTS


def _cut(tmp_path):
    folder = st.write_model(tmp_path / "st")
    return fse.Encoder.from_pretrained(folder, truncate_dim=4, normalize=True), TEXTS


def _words(_tmp_path):
    return finished(), TEXTS


def _fasttext(_tmp_path):
    return finished(fasttext=True), TEXTS


def _bits(rows):
    return rows.view(np.uint32).tobytes()


@pytest.mark.parametrize("make", [_wordpiece, _unigram, _cut, _words, _fasttext])
@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_a_quantized_encoder_is_saved_pushed_and_loaded_back(make, dtype, fake, tmp_path):
    encoder, texts = make(tmp_path)
    quantized = encoder.quantize(dtype)
    want = quantized.encode(texts, workers=1)
    metadata = quantized.word_vectors.metadata
    assert metadata["fse.quantized_from"] == encoder.word_vectors.fingerprint
    quantized.save(tmp_path / "q.safetensors")
    for mmap in (False, True):
        loaded = fse.Encoder.load(tmp_path / "q.safetensors", mmap=mmap)
        assert loaded.fingerprint == quantized.fingerprint
        for key in ("fse.dtype", "fse.quantization", "fse.quantized_from"):
            assert loaded.word_vectors.metadata[key] == metadata[key]
        assert _bits(loaded.encode(texts, workers=1)) == _bits(want)
        loaded.verify()
    folder = quantized.save_pretrained(tmp_path / "folder", license="mit")
    card = (folder / "README.md").read_text(encoding="utf-8")
    assert (
        f"- table stored by `quantize_rows@1` from the float32 table `{metadata['fse.quantized_from']}`"
        in card
    )
    assert "not promised by this file" in card
    assert f"({dtype})" in card
    for mmap in (False, True):
        local = fse.Encoder.from_pretrained(folder, mmap=mmap)
        assert local.fingerprint == quantized.fingerprint
        assert _bits(local.encode(texts, workers=1)) == _bits(want)
    quantized.push_to_hub(REPO, license="mit")
    commit = fake.pushes[-1]
    assert sorted(fake.commits[commit]) == [
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    for revision in (None, commit):
        pushed = fse.Encoder.from_pretrained(REPO, revision=revision, cache_dir=fake.cache)
        assert pushed.fingerprint == quantized.fingerprint
        assert pushed.word_vectors.dtype == dtype
        assert _bits(pushed.encode(texts, workers=1)) == _bits(want)
    assert pushed.fingerprint != encoder.fingerprint
