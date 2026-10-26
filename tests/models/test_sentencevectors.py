#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers


"""Automated tests for checking the sentence vectors."""

import numpy as np
import pytest
from gensim import matutils

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
        # Row 0 is unit length, so it must be the same in both arrays.
        assert np.allclose(sv.vectors[0], sv.vectors_norm[0])

    def test_init_sims_w_replace_overwrites_vectors(self, sv):
        expected = self._normalized(sv.vectors)
        sv.init_sims(True)
        assert np.allclose(sv.vectors, expected)

    def test_init_sims_w_mapfile(self, sv, tmp_path):
        sv.mapfile_path = str(tmp_path / "test_vectors")
        sv.init_sims()
        assert (tmp_path / "test_vectors.vectors_norm").exists()

    def test_init_sims_w_mapfile_path_writes_normalized_file(self, sv, tmp_path):
        """SentenceVectors stores mapfile_path as Path; the norms go into the mapfile."""
        expected = self._normalized(sv.vectors)
        sv.mapfile_path = tmp_path / "test_vectors"
        sv.init_sims()
        assert isinstance(sv.vectors_norm, np.memmap)
        on_disk = np.memmap(
            tmp_path / "test_vectors.vectors_norm", dtype=np.float32, mode="r", shape=expected.shape
        )
        assert np.allclose(on_disk, expected)

    def test_get_vector_use_norm_before_init_sims(self, sv):
        expected = self._normalized(sv.vectors)
        assert np.allclose(sv.get_vector(1, use_norm=True), expected[1])

    def test_init_sims_keeps_zero_vectors(self):
        sv = SentenceVectors(2)
        sv.vectors = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        sv.init_sims()
        assert np.array_equal(sv.vectors_norm, np.array([[0.0, 0.0], [0.6, 0.8]], np.float32))

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

    def test_most_similar_restrict_size_numpy_integer(self, trained):
        o = trained.sv.most_similar(positive=20, topn=20, restrict_size=np.int64(5))
        assert len(o) == 5

    def test_most_similar_restrict_size_reversed_raises(self, trained):
        with pytest.raises(ValueError, match="upper bound below its lower bound"):
            trained.sv.most_similar(positive=1, topn=20, restrict_size=(25, 5))

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


class TestSentenceVectorsLookups:
    def test_contains_rejects_invalid_indices(self, sv):
        assert 0 in sv
        assert -1 not in sv
        assert "x" not in sv
        assert 1.5 not in sv

    def test_get_vector_missing_index_message(self, sv):
        with pytest.raises(KeyError, match="index 99 not found"):
            sv.get_vector(99)


class TestMinSimilarity:
    """most_similar and the similar_by_* methods filter by a similarity threshold (#34)."""

    def test_keeps_only_results_at_or_above_the_threshold(self, trained):
        everything = trained.sv.most_similar(positive=0, topn=50)
        threshold = everything[10][1]
        filtered = trained.sv.most_similar(positive=0, topn=50, min_similarity=threshold)
        assert filtered == [item for item in everything if item[1] >= threshold]
        assert filtered
        assert all(item[1] >= threshold for item in filtered)

    def test_thresholds_at_the_bounds(self, trained):
        assert len(trained.sv.most_similar(positive=0, topn=5, min_similarity=-1)) == 5
        result = trained.sv.most_similar(positive=0, topn=5, min_similarity=1)
        assert all(item[1] >= 1 for item in result)

    def test_applies_to_indexable_results(self, trained, indexed_sentences):
        result = trained.sv.most_similar(
            positive=0, indexable=indexed_sentences, topn=20, min_similarity=0.5
        )
        assert all(len(item) == 3 and item[2] >= 0.5 for item in result)

    def test_similar_by_methods_pass_it_on(self, trained, w2v):
        vector = trained.sv[0]
        assert trained.sv.similar_by_vector(vector, topn=20, min_similarity=0.9) == (
            trained.sv.most_similar(positive=vector, topn=20, min_similarity=0.9)
        )
        by_word = trained.sv.similar_by_word("the", w2v.wv, topn=20, min_similarity=0.5)
        assert all(item[1] >= 0.5 for item in by_word)
        by_sentence = trained.sv.similar_by_sentence(
            ["the", "dog"], trained, topn=20, min_similarity=0.5
        )
        assert all(item[1] >= 0.5 for item in by_sentence)

    def test_accepts_numpy_floats(self, trained):
        assert trained.sv.most_similar(positive=0, topn=5, min_similarity=np.float32(0.0)) == (
            trained.sv.most_similar(positive=0, topn=5, min_similarity=0.0)
        )

    @pytest.mark.parametrize("value", [1.5, -1.01, float("nan"), float("inf"), "0.5"])
    def test_rejects_invalid_thresholds(self, trained, value):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            trained.sv.most_similar(positive=0, topn=5, min_similarity=value)

    @pytest.mark.parametrize("topn", [0, None])
    def test_requires_a_top_n_list(self, trained, topn):
        with pytest.raises(ValueError, match="needs a positive topn"):
            trained.sv.most_similar(positive=0, topn=topn, min_similarity=0.5)


def _reference_rows(vectors, query, count):
    """All dot products at once, sorted by similarity and then by row."""
    sims = np.dot(vectors, query)
    keys = np.where(np.isnan(sims), np.inf, -sims)
    order = np.lexsort((np.arange(len(sims)), keys))[:count]
    return order, sims[order]


class TestBlockwiseSearch:
    """most_similar keeps a running top-k over blocks of rows."""

    @pytest.fixture
    def vectors(self):
        # Small integers, so every dot product is exact in float32 whatever the BLAS and
        # the block size; the selection is then compared bit for bit. They give many ties.
        rng = np.random.default_rng(5)
        vectors = rng.integers(-4, 5, size=(200, 8)).astype(np.float32)
        # The best rows: exact ties across block boundaries.
        vectors[[3, 7, 50, 51, 120, 199]] = 10
        vectors[[10, 90]] = np.nan
        return vectors

    @pytest.mark.parametrize("block", [1, 7, 64, 1000])
    @pytest.mark.parametrize("count", [1, 3, 6, 25, 199, 500])
    def test_matches_a_full_sort(self, vectors, monkeypatch, block, count):
        from fse.models import sentencevectors

        monkeypatch.setattr(sentencevectors, "REMOVAL_BLOCK_ROWS", block)
        query = vectors[7]
        rows, sims = sentencevectors._most_similar_rows(vectors, query, count)
        expected_rows, expected_sims = _reference_rows(vectors, query, count)
        assert rows.tolist() == expected_rows.tolist()
        np.testing.assert_array_equal(sims, expected_sims)

    @pytest.mark.parametrize(("count", "largest"), [(5, 7), (30, 30), (300, 300)])
    def test_reads_blocks_of_bounded_size(self, vectors, monkeypatch, count, largest):
        """No dot product covers more rows than max(block, count)."""
        from fse.models import sentencevectors

        sizes = []

        def spy(rows, query):
            sizes.append(len(rows))
            return np.dot(rows, query)

        monkeypatch.setattr(sentencevectors, "REMOVAL_BLOCK_ROWS", 7)
        monkeypatch.setattr(sentencevectors, "dot", spy)
        rows, _ = sentencevectors._most_similar_rows(vectors, vectors[7], count)
        assert max(sizes) == min(largest, len(vectors))
        assert sum(sizes) == len(vectors)
        assert len(rows) == min(count, len(vectors))

    def test_ties_go_to_the_lower_index(self, vectors):
        from fse.models import sentencevectors

        rows, _ = sentencevectors._most_similar_rows(vectors, vectors[7], 6)
        assert rows.tolist() == [3, 7, 50, 51, 120, 199]

    @pytest.mark.parametrize("restrict_size", [None, 50, (20, 180)])
    def test_most_similar_does_not_depend_on_the_block_size(
        self, trained, monkeypatch, restrict_size
    ):
        from fse.models import sentencevectors

        def query():
            return trained.sv.most_similar(
                positive=[1, 4], negative=[2], topn=15, restrict_size=restrict_size
            )

        whole = query()
        monkeypatch.setattr(sentencevectors, "REMOVAL_BLOCK_ROWS", 3)
        # BLAS may round the dot products of a small block differently than of all rows.
        blocked = query()
        assert [index for index, _ in blocked] == [index for index, _ in whole]
        assert [sim for _, sim in blocked] == pytest.approx([sim for _, sim in whole], abs=1e-6)
        lo = 0 if not isinstance(restrict_size, tuple) else restrict_size[0]
        # The query vector exactly as most_similar builds it.
        norm = trained.sv.vectors_norm
        mean = matutils.unitvec(np.array([norm[1], norm[4], -norm[2]]).mean(axis=0))
        mean = mean.astype(np.float32)
        hi = {None: None, 50: 50}.get(restrict_size, 180)
        sims = np.dot(trained.sv.vectors_norm[lo:hi], mean)
        for index, sim in whole:
            assert sim == pytest.approx(float(sims[index - lo]), abs=1e-6)
            assert index not in (1, 2, 4)
