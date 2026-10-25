#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Automated tests for checking the average model."""

import numpy as np
import pytest

from fse.models import average
from fse.models.average import Average, train_average_np
from fse.models.average_inner import (
    FAST_VERSION,
    MAX_NGRAMS_IN_BATCH,
    MAX_WORDS_IN_BATCH,
    train_average_cy,
)
from fse.models.base_s2v import EPS

DIM = 5


@pytest.fixture
def w2v(make_w2v):
    """Word2Vec model whose vector for each word is its vocabulary index."""
    w2v = make_w2v(DIM)
    w2v.wv.vectors[:,] = np.arange(len(w2v.wv.vectors), dtype=np.float32)[:, None]
    return w2v


@pytest.fixture
def ft(make_ft):
    return make_ft(DIM)


@pytest.fixture
def indexed_sentences():
    sentences = [
        ["They", "admit"],
        ["So", "Apple", "bought", "buds"],
        ["go", "12345"],
        ["pull", "12345678910111213"],
    ]
    return [(s, i) for i, s in enumerate(sentences)]


@pytest.fixture
def corpus_indexed(sentences):
    return [(s, i) for i, s in enumerate(sentences)]


def _prepare(model, total_sentences):
    model.prep.prepare_vectors(sv=model.sv, total_sentences=total_sentences, update=False)
    model._pre_train_calls()
    return model


@pytest.fixture
def model(w2v, indexed_sentences):
    return _prepare(Average(w2v), len(indexed_sentences))


def _mean_index(w2v, words):
    """Expected average for w2v, whose vector for each word is its vocabulary index.

    The index of words with equal counts depends on the gensim version, so the
    expectation is derived from key_to_index instead of being hard coded.
    """
    return np.mean([w2v.wv.key_to_index[w] for w in words])


def test_cython():
    assert FAST_VERSION
    assert MAX_WORDS_IN_BATCH == 10000
    assert MAX_NGRAMS_IN_BATCH == 40


def test_fast_version_loaded():
    """The compiled extension must be importable, not the numpy fallback."""
    assert average.FAST_VERSION >= 0


@pytest.mark.parametrize("train_func", [train_average_np, train_average_cy], ids=["np", "cy"])
def test_average_train_w2v(model, w2v, indexed_sentences, train_func):
    model.sv.vectors = np.zeros_like(model.sv.vectors, dtype=np.float32)
    mem = model._get_thread_working_mem()
    output = train_func(model, indexed_sentences, model.sv.vectors, mem)
    assert output == (4, 7)
    assert np.allclose(_mean_index(w2v, ["They", "admit"]), model.sv[0])
    assert np.allclose(_mean_index(w2v, ["So", "Apple", "bought", "buds"]), model.sv[1])
    assert (model.wv.key_to_index["go"] == model.sv[2]).all()


@pytest.mark.parametrize("train_func", [train_average_np, train_average_cy], ids=["np", "cy"])
def test_average_train_ft(ft, indexed_sentences, train_func):
    m = _prepare(Average(ft), len(indexed_sentences))
    m.wv.vectors = m.wv.vectors_vocab = np.ones_like(m.wv.vectors, dtype=np.float32)
    m.wv.vectors_ngrams = np.full_like(m.wv.vectors_ngrams, 2, dtype=np.float32)
    mem = m._get_thread_working_mem()
    output = train_func(m, indexed_sentences, m.sv.vectors, mem)
    assert output == (4, 10)
    assert (m.sv[0] == 1.0 + EPS).all()
    # "go" -> [1,1...]
    # oov: "12345" -> (14 hashes * 2) / 14 =  2
    # (2 + 1) / 2 = 1.5
    assert np.allclose(1.5, m.sv[2])
    assert np.allclose(2, m.sv[3])


def test_cy_equal_np_w2v(w2v, indexed_sentences):
    m1 = _prepare(Average(w2v), len(indexed_sentences))
    mem1 = m1._get_thread_working_mem()
    o1 = train_average_np(m1, indexed_sentences, m1.sv.vectors, mem1)

    m2 = _prepare(Average(w2v), len(indexed_sentences))
    mem2 = m2._get_thread_working_mem()
    o2 = train_average_cy(m2, indexed_sentences, m2.sv.vectors, mem2)

    assert o1 == o2
    assert (m1.sv.vectors == m2.sv.vectors).all()


def _train_many_to_one(model, indexed_sentences, train_func):
    """Train sentences that share target rows (n:m mapping)."""
    sentences = [(s, i // 2) for i, (s, _) in enumerate(indexed_sentences)]
    _prepare(model, 2)
    model.sv.vectors = np.zeros_like(model.sv.vectors, dtype=np.float32)
    mem = model._get_thread_working_mem()
    output = train_func(model, sentences, model.sv.vectors, mem)
    return output, model.sv.vectors[:2]


def test_cy_equal_np_w2v_many_to_one(w2v, indexed_sentences):
    o1, v1 = _train_many_to_one(Average(w2v), indexed_sentences, train_average_np)
    o2, v2 = _train_many_to_one(Average(w2v), indexed_sentences, train_average_cy)
    assert o1 == o2
    assert np.allclose(v1, v2)
    expected = _mean_index(w2v, ["They", "admit"]) + _mean_index(
        w2v, ["So", "Apple", "bought", "buds"]
    )
    assert np.allclose(expected, v1[0])


def test_cy_equal_np_ft_many_to_one(ft, indexed_sentences):
    m1 = Average(ft)
    m1.batch_ngrams = MAX_NGRAMS_IN_BATCH
    o1, v1 = _train_many_to_one(m1, indexed_sentences, train_average_np)
    o2, v2 = _train_many_to_one(Average(ft), indexed_sentences, train_average_cy)
    assert o1 == o2
    assert np.allclose(v1, v2, atol=1e-6)


def test_cy_equal_np_w2v_random(make_w2v, indexed_sentences):
    # Random initialization
    w2v = make_w2v(DIM)

    m1 = _prepare(Average(w2v), len(indexed_sentences))
    mem1 = m1._get_thread_working_mem()
    train_average_np(m1, indexed_sentences, m1.sv.vectors, mem1)

    m2 = _prepare(Average(w2v), len(indexed_sentences))
    mem2 = m2._get_thread_working_mem()
    train_average_cy(m2, indexed_sentences, m2.sv.vectors, mem2)

    assert np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6)


def test_cy_equal_np_ft_random(make_ft, indexed_sentences):
    ft = make_ft(20)

    m1 = _prepare(Average(ft), len(indexed_sentences))
    m1.batch_ngrams = MAX_NGRAMS_IN_BATCH
    mem1 = m1._get_thread_working_mem()
    o1 = train_average_np(m1, indexed_sentences[:2], m1.sv.vectors, mem1)

    m2 = _prepare(Average(ft), len(indexed_sentences))
    mem2 = m2._get_thread_working_mem()
    o2 = train_average_cy(m2, indexed_sentences[:2], m2.sv.vectors, mem2)

    assert o1 == o2
    assert np.allclose(m1.sv.vectors, m2.sv.vectors, atol=1e-6)


def test_do_train_job(model, sentences, corpus_indexed):
    model.prep.prepare_vectors(sv=model.sv, total_sentences=len(sentences), update=True)
    mem = model._get_thread_working_mem()
    output = model._do_train_job(corpus_indexed, target=model.sv.vectors, memory=mem)
    assert output == (100, 1450)
    assert model.sv.vectors.shape == (104, DIM)


def test_train(model, corpus_indexed):
    assert model.train(corpus_indexed) == (100, 1450)


@pytest.mark.parametrize("workers", [1, 2], ids=["single", "multi"])
def test_train_from_disk(w2v, corpus_indexed, tmp_path, workers):
    """Training with vectors mapped to disk gives the same result as in memory."""
    p = tmp_path / "test_vecs"
    p_target = tmp_path / "test_vecs_wv.vectors"

    se1 = Average(w2v, workers=workers)
    se2 = Average(w2v, workers=workers, sv_mapfile_path=str(p), wv_mapfile_path=str(p))
    se1.train(corpus_indexed)
    se2.train(corpus_indexed)

    assert (tmp_path / "test_vecs.vectors").exists()
    assert p_target.exists()
    assert (se1.wv.vectors == se2.wv.vectors).all()
    assert not se2.wv.vectors.flags.writeable

    assert (se1.sv.vectors == se2.sv.vectors).all()


def test_check_parameter_sanity(w2v):
    se = Average(w2v)
    se.word_weights = np.full(20, 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="word weights must equal one"):
        se._check_parameter_sanity()


def test_train_with_numpy_custom_index(w2v, sentences):
    """CIndexedList documents ndarray custom indices; training must accept them."""
    from fse import CIndexedList

    custom_index = np.arange(len(sentences), dtype=np.int64) // 2
    model = Average(w2v)
    model.train(CIndexedList(sentences, custom_index=custom_index))
    assert model.sv.vectors.shape == (len(sentences) // 2, w2v.wv.vector_size)

    expected = Average(w2v)
    expected.train(CIndexedList(sentences, custom_index=custom_index.tolist()))
    assert np.allclose(model.sv.vectors, expected.sv.vectors)


class TestKernelInputChecks:
    """train_average_cy takes raw pointers, so it must reject arrays it cannot use."""

    def _model(self, w2v):
        model = Average(w2v)
        _prepare(model, total_sentences=2)
        return model

    def test_rejects_wrong_dtype_target(self, w2v):
        model = self._model(w2v)
        target = np.zeros(model.sv.vectors.shape, dtype=np.float64)
        with pytest.raises(TypeError, match="target must have dtype float32"):
            train_average_cy(model, [(["the"], 0)], target, model._get_thread_working_mem())

    def test_rejects_non_contiguous_target(self, w2v):
        model = self._model(w2v)
        target = np.zeros((model.sv.vectors.shape[0], 2 * w2v.wv.vector_size), np.float32)
        with pytest.raises(TypeError, match="C-contiguous"):
            train_average_cy(model, [(["the"], 0)], target[:, ::2], model._get_thread_working_mem())

    def test_rejects_index_out_of_range(self, w2v):
        model = self._model(w2v)
        with pytest.raises(IndexError, match="out of range"):
            train_average_cy(
                model, [(["the"], 2)], model.sv.vectors, model._get_thread_working_mem()
            )

    def test_rejects_small_ngram_buffer(self, ft):
        model = Average(ft)
        _prepare(model, total_sentences=1)
        mem, _ = model._get_thread_working_mem()
        small = np.zeros((2, 2), dtype=np.uint32)
        with pytest.raises(ValueError, match="memory\\[1\\] must hold"):
            train_average_cy(model, [(["the"], 0)], model.sv.vectors, (mem, small))

    def test_small_batch_settings_keep_full_ngram_buffer(self, ft):
        """batch_words and batch_ngrams below the kernel limits must not shrink the buffer."""
        model = Average(ft)
        model.batch_words, model.batch_ngrams = 2, 2
        _, oov_mem = model._get_thread_working_mem()
        assert oov_mem.shape == (MAX_WORDS_IN_BATCH, MAX_NGRAMS_IN_BATCH)


def test_most_similar_with_sentence_mapfile(w2v, corpus_indexed, tmp_path):
    """most_similar normalizes into a mapfile when the sentence vectors live on disk."""
    on_disk = Average(w2v, sv_mapfile_path=str(tmp_path / "sv"))
    on_disk.train(corpus_indexed)
    in_ram = Average(w2v)
    in_ram.train(corpus_indexed)
    assert on_disk.sv.most_similar(positive=0, topn=5) == in_ram.sv.most_similar(positive=0, topn=5)
    assert (tmp_path / "sv.vectors_norm").exists()


def test_infer_use_norm_with_unknown_words(w2v, corpus_indexed):
    """A sentence without known words has a zero vector, not NaN, after normalization."""
    model = Average(w2v)
    model.train(corpus_indexed)
    known = next(w for w in w2v.wv.index_to_key if w2v.wv.key_to_index[w] > 0)
    out = model.infer([(["unknownword"], 0), ([known], 1)], use_norm=True)
    assert np.array_equal(out[0], np.zeros(w2v.wv.vector_size, dtype=np.float32))
    assert np.isclose(np.linalg.norm(out[1]), 1.0)
