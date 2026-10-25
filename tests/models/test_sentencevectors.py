#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers


"""Automated tests for checking the sentence vectors."""

import numpy as np
import pytest

from fse.inputs import IndexedLineDocument, IndexedList
from fse.models.average import Average
from fse.models.sentencevectors import SentenceVectors, _l2_norm

DIM = 5


@pytest.fixture
def w2v(make_w2v):
    """Word2Vec model whose vectors are random but independent of the vocabulary order.

    Each word gets its random vector by alphabetical rank, so the result does not
    depend on how a gensim version orders words with equal counts.
    """
    w2v = make_w2v(DIM, seed=42)
    rng = np.random.default_rng(12345)
    values = rng.uniform(size=w2v.wv.vectors.shape).astype(np.float32)
    rank = {word: rank for rank, word in enumerate(sorted(w2v.wv.index_to_key))}
    w2v.wv.vectors = values[[rank[word] for word in w2v.wv.index_to_key]]
    return w2v


@pytest.fixture
def sv():
    sv = SentenceVectors(2)
    sv.vectors = np.arange(10).reshape(5, 2)
    return sv


@pytest.fixture
def indexed_sentences(corpus_path):
    return IndexedLineDocument(corpus_path)


@pytest.fixture
def trained(w2v, indexed_sentences):
    """Average model trained on the corpus."""
    m = Average(w2v)
    m.train(indexed_sentences)
    return m


class TestSentenceVectors:
    def test_getitem(self, sv):
        assert (sv[0] == [0, 1]).all()
        assert (sv[[0, 2]] == [[0, 1], [4, 5]]).all()

    def test_isin(self, sv):
        assert 0 in sv
        assert 5 not in sv

    def test_init_sims_wo_replace(self, sv):
        sv.init_sims()
        assert sv.vectors_norm is not None
        assert not (sv.vectors == sv.vectors_norm).all()

        v1 = sv.vectors[0]
        v1 = v1 / np.sqrt(np.sum(v1**2))

        v2 = sv.vectors[1]
        v2 = v2 / np.sqrt(np.sum(v2**2))

        assert np.allclose(v1, sv.vectors_norm[0])
        assert np.allclose(v2, sv.vectors_norm[1])
        assert np.allclose(v2, sv.get_vector(1, True))

    def test_get_vector(self, sv):
        assert (sv.get_vector(0) == [0, 1]).all()
        assert (sv.get_vector(1) == [2, 3]).all()

    @staticmethod
    def _normalized(vectors):
        """Row-wise L2 normalization, checked to differ from the input.

        Row 0 of the fixture is [0, 1] and already has unit length, so a check on
        that row alone cannot tell normalized from raw vectors. The other rows can.
        """
        original = vectors.astype(np.float64)
        expected = original / np.linalg.norm(original, axis=1, keepdims=True)
        assert not np.allclose(original[1:], expected[1:])
        return expected

    def test_init_sims_w_replace(self, sv):
        expected = self._normalized(sv.vectors)
        sv.init_sims(True)
        assert np.allclose(sv.vectors_norm, expected)
        assert np.allclose(np.linalg.norm(sv.vectors_norm, axis=1), 1.0)

    @pytest.mark.xfail(
        reason="init_sims(replace=True) stores the normalized copy in vectors_norm but "
        "leaves vectors unchanged, because _l2_norm(replace=True) does not work in place",
    )
    def test_init_sims_w_replace_overwrites_vectors(self, sv):
        expected = self._normalized(sv.vectors)
        sv.init_sims(True)
        assert np.allclose(sv.vectors, expected)

    def test_init_sims_w_mapfile(self, sv, tmp_path):
        sv.mapfile_path = str(tmp_path / "test_vectors")
        sv.init_sims()
        assert (tmp_path / "test_vectors.vectors_norm").exists()

    def test_save_load(self, sv, tmp_path):
        p = tmp_path / "test_vectors.vectors"
        sv.save(str(p))
        assert p.exists()
        sv2 = SentenceVectors.load(str(p))
        assert (sv.vectors == sv2.vectors).all()

    def test_save_load_with_memmap(self, tmp_path):
        p = tmp_path / "test_vectors"
        p_target = tmp_path / "test_vectors.vectors"
        p_not_exists = tmp_path / "test_vectors.vectors.npy"

        sv = SentenceVectors(2, mapfile_path=str(p))

        shape = (1000, 1000)
        sv.vectors = np.ones(shape, dtype=np.float32)

        memvecs = np.memmap(p_target, dtype=np.float32, mode="w+", shape=shape)
        memvecs[:] = sv.vectors[:]
        del memvecs

        assert p_target.exists()
        sv.save(str(p))
        assert p.exists()
        assert not p_not_exists.exists()

        sv = SentenceVectors.load(str(p))
        assert sv.vectors.shape == shape

    def test_len(self, sv):
        assert len(sv) == 5

    def test_similarity(self, sv):
        v1 = sv.vectors[0]
        v1 = v1 / np.sqrt(np.sum(v1**2))

        v2 = sv.vectors[1]
        v2 = v2 / np.sqrt(np.sum(v2**2))

        assert np.allclose(v1.dot(v2), sv.similarity(0, 1))
        assert np.allclose(1 - v1.dot(v2), sv.distance(0, 1))


class TestMostSimilar:
    def test_most_similar(self, trained, indexed_sentences, sentences, corpus_path):
        sent_ind = IndexedList(sentences)
        indexable = indexed_sentences
        o = trained.sv.most_similar(positive=0)
        assert o[0][0] == 86
        assert o[1][0] == 66
        expected = corpus_path.read_text().splitlines()[86].rstrip()
        o = trained.sv.most_similar(positive=0, indexable=indexable)
        assert o[0][0] == expected

        o = trained.sv.most_similar(positive=0, indexable=sent_ind)
        assert o[0][0][0] == expected.split()

    def test_most_similar_vec(self, trained):
        trained.sv.init_sims()
        v = trained.sv.get_vector(0, use_norm=True)
        o = trained.sv.most_similar(positive=v)
        assert o[0][0] == 0
        assert o[1][0] == 86
        assert o[2][0] == 66

    def test_most_similar_vectors(self, trained):
        trained.sv.init_sims()
        v = trained.sv[[0, 1]]
        o = trained.sv.most_similar(positive=v)
        assert o[0][0] == 78
        assert o[1][0] == 53

    def test_most_similar_wrong_indexable(self, trained):
        def indexable(self):
            pass

        with pytest.raises(RuntimeError):
            trained.sv.most_similar(positive=0, indexable=indexable)

    def test_most_similar_topn(self, trained):
        o = trained.sv.most_similar(positive=0, topn=20)
        assert len(o) == 20

    def test_most_similar_restrict_size(self, trained):
        o = trained.sv.most_similar(positive=20, topn=20, restrict_size=5)
        assert len(o) == 5

    def test_most_similar_restrict_size_tuple(self, trained, indexed_sentences):
        o = trained.sv.most_similar(positive=20, topn=20, restrict_size=(5, 25))
        assert len(o) == 19
        assert o[0][0] == 7

        o = trained.sv.most_similar(positive=1, topn=20, restrict_size=(5, 25))
        assert len(o) == 20
        assert o[0][0] == 8

        o = trained.sv.most_similar(
            positive=1, topn=20, restrict_size=(5, 25), indexable=indexed_sentences
        )
        assert len(o) == 20
        assert o[0][1] == 8

    def test_similar_by_word(self, trained, indexed_sentences):
        o = trained.sv.similar_by_word(word="the", wv=trained.wv)
        assert o[0][0] == 55
        o = trained.sv.similar_by_word(word="the", wv=trained.wv, indexable=indexed_sentences)
        assert o[0][1] == 55

    def test_similar_by_vector(self, trained):
        o = trained.sv.similar_by_vector(trained.wv["the"])
        assert o[0][0] == 55

    def test_similar_by_sentence(self, trained):
        o = trained.sv.similar_by_sentence(sentence=["the", "product", "is", "good"], model=trained)
        assert o[0][0] == 83

    def test_similar_by_sentence_wrong_model(self, trained, w2v):
        with pytest.raises(RuntimeError):
            trained.sv.similar_by_sentence(sentence=["the", "product", "is", "good"], model=w2v)


def test_l2_norm():
    out = np.random.normal(size=(200, 50)).astype(np.float32)
    result = _l2_norm(out, False)
    lens = np.sqrt(np.sum((result**2), axis=-1))
    assert np.allclose(1, lens, atol=1e-6)

    out = np.random.normal(size=(200, 50)).astype(np.float32)
    out = _l2_norm(out, True)
    lens = np.sqrt(np.sum((out**2), axis=-1))
    assert np.allclose(1, lens, atol=1e-6)
