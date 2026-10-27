import sys

import numpy as np
import pytest
from gensim.models import KeyedVectors

from fse._reference import quantize_rows
from fse.gensim import from_gensim


def _kv(words, dim=4, seed=0):
    kv = KeyedVectors(dim)
    rows = np.random.default_rng(seed).standard_normal((len(words), dim)).astype(np.float32)
    kv.add_vectors(words, rows)
    for rank, word in enumerate(words, start=1):
        kv.set_vecattr(word, "count", 1000 // rank)
    return kv


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_keyed_vectors(dtype):
    kv = _kv(["the", "é", "naïve", "日本", ""])
    wv = from_gensim(kv, dtype)
    assert wv.dtype == dtype
    assert len(wv) == len(kv)
    for word, i in kv.key_to_index.items():
        assert wv.index(word) == i
        assert wv.word(i) == word
    codes, scales = quantize_rows(kv.vectors, dtype)
    assert wv.vectors.tobytes() == codes.tobytes()
    if dtype == "int8":
        assert wv.scales.tobytes() == scales.tobytes()
    np.testing.assert_array_equal(wv.counts, [kv.get_vecattr(w, "count") for w in kv.index_to_key])
    assert wv.fasttext is None
    # The words' rows are copied: changing gensim's vectors afterwards changes nothing.
    before = wv.get_vector(0).copy()
    kv.vectors[0] += 1
    np.testing.assert_array_equal(wv.get_vector(0), before)


def test_word2vec_and_counts(make_w2v):
    w2v = make_w2v(vector_size=6)
    wv = from_gensim(w2v)
    assert len(wv) == len(w2v.wv)
    for word, i in w2v.wv.key_to_index.items():
        np.testing.assert_array_equal(wv.get_vector(word), w2v.wv.vectors[i])
        assert wv.counts[i] == w2v.wv.get_vecattr(word, "count")
    assert from_gensim(w2v, counts=False).counts is None


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_fasttext(make_ft, dtype):
    ft = make_ft(vector_size=6, bucket=100, min_n=2, max_n=4)
    wv = from_gensim(ft, dtype)
    params = wv.fasttext
    assert (params.min_n, params.max_n, params.bucket) == (2, 4, 100)
    codes, _ = quantize_rows(ft.wv.vectors_vocab, dtype)
    assert wv.vectors.tobytes() == codes.tobytes()
    ngram_codes, ngram_scales = quantize_rows(ft.wv.vectors_ngrams, dtype)
    assert wv.ngrams.tobytes() == ngram_codes.tobytes()
    if dtype == "int8":
        assert wv.ngram_scales.tobytes() == ngram_scales.tobytes()
    assert from_gensim(ft.wv).fingerprint == from_gensim(ft).fingerprint


def test_save_and_load(tmp_path, make_ft):
    from fse import WordVectors

    wv = from_gensim(make_ft(vector_size=4, bucket=50), "int8")
    wv.save(tmp_path / "wv")
    assert WordVectors.load(tmp_path / "wv", mmap=True).fingerprint == wv.fingerprint


def test_rejects():
    with pytest.raises(TypeError, match="KeyedVectors"):
        from_gensim(np.ones((2, 2)))
    kv = _kv(["a", "b"])
    kv.vectors = kv.vectors.astype(np.float64)
    with pytest.raises(TypeError, match="must be float32"):
        from_gensim(kv)
    kv = _kv(["a", "b"])
    kv.index_to_key.append("c")
    with pytest.raises(ValueError, match="holds 3 words"):
        from_gensim(kv)
    kv = _kv(["a", "\ud800"])
    with pytest.raises(ValueError, match=r"word 1 .* has no UTF-8 form"):
        from_gensim(kv)
    kv = _kv(["a", 2])
    with pytest.raises(TypeError, match="word 1 must be a str"):
        from_gensim(kv)
    kv = _kv(["a", "b"])
    kv.vectors[1, 0] = np.nan
    with pytest.raises(ValueError, match="word vector 1 contains NaN"):
        from_gensim(kv, "int8")
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="int8"):
        from_gensim(_kv(["a"]), "int4")


def test_fasttext_rejects(make_ft):
    ft = make_ft(vector_size=4, bucket=50)
    ft.wv.compatible_hash = False
    with pytest.raises(ValueError, match="incompatible ngram hash"):
        from_gensim(ft)
    ft = make_ft(vector_size=4, bucket=50)
    ft.wv.bucket = 0
    with pytest.raises(ValueError, match="without ngram buckets"):
        from_gensim(ft)


def test_missing_gensim(monkeypatch):
    monkeypatch.setitem(sys.modules, "gensim", None)
    with pytest.raises(ImportError, match=r"pip install 'fse\[gensim\]'"):
        from_gensim(object())


def test_fse_gensim_is_the_adapter():
    from fse import gensim

    assert gensim.from_gensim is from_gensim
