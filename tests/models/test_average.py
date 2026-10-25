#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Automated tests for checking the average model."""

import sys

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

    def test_working_memory_has_no_ngram_buffer(self, ft):
        """The kernel allocates the ngram buffer per job, sized by batch_ngrams."""
        _, oov_mem = Average(ft)._get_thread_working_mem()
        assert oov_mem.size == 0


@pytest.mark.parametrize("kind", ["w2v", "ft"])
def test_cy_averages_sentences_longer_than_10000_words_like_np(kind, w2v, ft):
    """The kernel used to stop after 10,000 words per sentence; numpy never did."""
    model = Average(w2v if kind == "w2v" else ft)
    words = list(model.wv.index_to_key)
    sentence = [words[i % len(words)] for i in range(25_003)] + ["unknownword"] * 7
    outputs = []
    for train_func in (train_average_np, train_average_cy):
        _prepare(model, total_sentences=1)
        counts = train_func(
            model, [(sentence, 0)], model.sv.vectors, model._get_thread_working_mem()
        )
        outputs.append((counts, np.array(model.sv.vectors)))
    assert outputs[0][0] == outputs[1][0]
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("batch_ngrams", [1, 2, 40, 500])
def test_cy_honors_batch_ngrams_like_np(ft, batch_ngrams):
    """Unknown FastText words use at most batch_ngrams ngrams in both code paths."""
    model = Average(ft)
    model.batch_ngrams = batch_ngrams
    sentences = [(["thisisaverylongunknownword", "the", "xyz"], 0), (["ünbekanntes"], 1)]
    outputs = []
    for train_func in (train_average_np, train_average_cy):
        _prepare(model, total_sentences=2)
        train_func(model, sentences, model.sv.vectors, model._get_thread_working_mem())
        outputs.append(np.array(model.sv.vectors))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)


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


@pytest.mark.parametrize("model_name", ["Average", "SIF", "uSIF"])
def test_train_update_appends_rows(make_w2v, sentences, model_name):
    """update=True writes new sentences behind the existing rows and keeps the old ones.

    The new sentences use indices 0..n-1 like any input; fse places them after the
    existing rows. SIF and uSIF keep the components learned in the first training, so
    the new rows equal infer() on the new sentences.
    """
    import fse

    # Random vectors: the module's row-index vectors are all parallel, which leaves only
    # rounding noise once SIF removes their common direction.
    w2v = make_w2v(DIM)
    w2v.wv.vectors[:] = np.random.default_rng(0).random(w2v.wv.vectors.shape, dtype=np.float32)
    first = [(s, i) for i, s in enumerate(sentences[:60])]
    second = [(s, i) for i, s in enumerate(sentences[60:])]
    model = getattr(fse, model_name)(w2v)
    model.train(first)
    old_rows = np.array(model.sv.vectors)
    svd_before = getattr(model, "svd_res", None)
    weights_before = getattr(model, "svd_weights", None)

    model.train(second, update=True)

    # The components and weights of the first training are reused, not recomputed.
    assert getattr(model, "svd_res", None) is svd_before
    assert getattr(model, "svd_weights", None) is weights_before

    assert model.sv.vectors.shape == (100, w2v.wv.vector_size)
    assert np.array_equal(model.sv.vectors[:60], old_rows)

    # Trained rows start at EPS while infer starts at zero. Component removal is linear,
    # so a trained row equals the inferred row plus the processed EPS offset.
    offset = np.full((1, w2v.wv.vector_size), EPS, dtype=np.float32)
    if model_name != "Average":
        model._post_inference_calls(output=offset)
    # float32 with a different summation order per BLAS: values near zero differ by about 1e-8.
    assert np.allclose(model.sv.vectors[60:], model.infer(second) + offset, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("model_name", ["SIF", "uSIF"])
def test_train_twice_without_components(w2v, corpus_indexed, model_name):
    """components=0 stores the marker 0 as svd_res; a second training must still work."""
    import fse

    model = getattr(fse, model_name)(w2v, components=0)
    model.train(corpus_indexed)
    first = np.array(model.sv.vectors)
    model.train(corpus_indexed)
    assert np.allclose(model.sv.vectors, first)


@pytest.mark.parametrize("token", ["", "§"])
def test_ft_oov_without_ngrams_parity(make_ft, token):
    """An OOV token without ngrams adds nothing in both paths but counts for the length.

    With min_n=4 the token "§" ("<§>" has 3 characters) and the empty token have no ngrams.
    Neither occurs in the test corpus.
    """
    ft = make_ft(5, min_n=4, max_n=4)
    assert token not in ft.wv.key_to_index
    sentences = [([token, "the"], 0), ([token], 1)]
    results = []
    for train_func in (train_average_np, train_average_cy):
        model = Average(ft)
        _prepare(model, total_sentences=2)
        model.sv.vectors = np.zeros_like(model.sv.vectors)
        train_func(model, sentences, model.sv.vectors, model._get_thread_working_mem())
        results.append(np.array(model.sv.vectors))
    np_out, cy_out = results
    assert np.allclose(np_out, cy_out, atol=1e-6)
    assert np.array_equal(cy_out[1], np.zeros(5, dtype=np.float32))


def test_save_load_with_sentence_mapfile(w2v, corpus_indexed, tmp_path):
    """Sentence vectors on disk are not duplicated into .npy files and are reopened by load."""
    model = Average(w2v, sv_mapfile_path=str(tmp_path / "sv"))
    model.train(corpus_indexed)
    model.sv.init_sims()
    expected = np.array(model.sv.vectors)

    # gensim stores arrays above sep_limit (10 MB) as separate .npy files; force that here.
    model.save(str(tmp_path / "model"), sep_limit=0)
    assert not [f.name for f in tmp_path.iterdir() if f.name.startswith("model.sv.")]
    assert isinstance(model.sv.vectors, np.memmap)

    loaded = Average.load(str(tmp_path / "model"))
    assert isinstance(loaded.sv.vectors, np.memmap)
    assert loaded.sv.vectors.filename == str((tmp_path / "sv.vectors").absolute())
    assert np.array_equal(loaded.sv.vectors, expected)
    assert loaded.sv.most_similar(positive=0, topn=3) == model.sv.most_similar(positive=0, topn=3)


def test_np_counts_empty_sentences_like_cy(w2v):
    sentences = [([], 0), (["the"], 1), ([], 2)]
    counts = []
    for train_func in (train_average_np, train_average_cy):
        model = Average(w2v)
        _prepare(model, total_sentences=3)
        counts.append(
            train_func(model, sentences, model.sv.vectors, model._get_thread_working_mem())
        )
    assert counts[0] == counts[1] == (1, 1)


def test_many_to_one_with_workers_loses_no_updates():
    """Workers that add into the same rows concurrently must not lose sentence averages.

    50,000 sentences map onto 4 rows, so each row sums 12,500 averages. One lost update
    changes a row by about 1/12,500 = 8e-5, far above float32 summation noise.
    """
    from gensim.models import KeyedVectors

    from fse import CIndexedList

    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(2000)]
    kv = KeyedVectors(16)
    kv.add_vectors(vocab, rng.random((len(vocab), 16), dtype=np.float32))
    for word in vocab:
        kv.set_vecattr(word, "count", 1)
    sentences = [[vocab[j] for j in rng.integers(0, len(vocab), 12)] for _ in range(50_000)]
    index = [i % 4 for i in range(len(sentences))]

    vectors = kv.vectors.astype(np.float64)
    expected = np.full((4, 16), EPS, dtype=np.float64)
    for sentence, row in zip(sentences, index, strict=True):
        expected[row] += vectors[[kv.key_to_index[w] for w in sentence]].mean(axis=0)

    for _ in range(5):
        model = Average(kv, workers=8)
        model.train(CIndexedList(sentences, custom_index=index))
        assert np.allclose(model.sv.vectors, expected, rtol=1e-5, atol=0)


def test_load_keeps_sentence_vectors_of_older_saves(w2v, corpus_indexed, tmp_path):
    """Before 1.1.0, a model save pickled memmapped sentence vectors without their shape."""
    from gensim.utils import SaveLoad

    model = Average(w2v, sv_mapfile_path=str(tmp_path / "sv"))
    model.train(corpus_indexed)
    expected = np.array(model.sv.vectors)
    model.sv.mapfile_shape = None
    # Save the way 1.0.0 did: through gensim, with the vectors inside the saved model.
    SaveLoad.save(model, str(tmp_path / "old_model"))

    loaded = Average.load(str(tmp_path / "old_model"))
    assert loaded.sv.vectors.shape == expected.shape
    assert np.array_equal(loaded.sv.vectors, expected)


class TestKernelShapeChecks:
    def _model(self, w2v):
        model = Average(w2v)
        _prepare(model, total_sentences=2)
        return model

    def test_rejects_read_only_target(self, w2v):
        model = self._model(w2v)
        target = np.zeros_like(model.sv.vectors)
        target.setflags(write=False)
        with pytest.raises(ValueError, match="writeable"):
            train_average_cy(model, [(["the"], 0)], target, model._get_thread_working_mem())

    def test_rejects_missing_word_weights(self, w2v):
        model = self._model(w2v)
        model.word_weights = np.zeros(0, dtype=np.float32)
        with pytest.raises(ValueError, match="one weight per word vector"):
            train_average_cy(
                model, [(["the"], 0)], model.sv.vectors, model._get_thread_working_mem()
            )

    def test_rejects_mismatched_vector_size(self, w2v):
        model = self._model(w2v)
        model.sv.vector_size = w2v.wv.vector_size + 1
        target = np.zeros((2, w2v.wv.vector_size + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="word vectors must have shape"):
            train_average_cy(model, [(["the"], 0)], target, model._get_thread_working_mem())


@pytest.mark.parametrize(("index", "expect_locks"), [(lambda i: i % 4, True), (lambda i: i, False)])
def test_row_locks_are_set_only_for_shared_rows(w2v, sentences, index, expect_locks):
    """Shared rows with several workers must train under row locks; unique rows without."""
    model = Average(w2v, workers=2)
    seen = []
    original = model._do_train_job

    def recording_job(data_iterable, target, memory):
        seen.append(getattr(model, "_row_locks", None) is not None)
        return original(data_iterable, target, memory)

    model._do_train_job = recording_job
    model.train([(s, index(i)) for i, s in enumerate(sentences)])
    assert seen
    assert all(flag is expect_locks for flag in seen)


@pytest.mark.skipif(sys.platform == "win32", reason="NTFS does not create sparse memmap files")
def test_cy_addresses_rows_beyond_32_bit_offsets(make_w2v, tmp_path):
    """Row * dimension exceeds 2**32 from row 14,316,558 on at 300 dimensions.

    The kernel computed that flat offset in uint32, so the sentence landed near the start
    of the matrix. The memmap is a sparse file; only the touched pages use disk space.
    """
    dim = 300
    rows = 2**32 // dim + 2
    model = Average(make_w2v(dim))
    model._pre_train_calls()
    target = np.memmap(tmp_path / "big", dtype=np.float32, mode="w+", shape=(rows, dim))
    word = model.wv.index_to_key[0]

    train_average_cy(model, [([word], rows - 1)], target, model._get_thread_working_mem())

    np.testing.assert_allclose(target[rows - 1], model.wv.vectors[0], rtol=1e-6)
    wrapped = (rows - 1) * dim % 2**32 // dim
    assert not target[: wrapped + 2].any()


@pytest.mark.parametrize("model_name", ["Average", "SIF", "uSIF"])
@pytest.mark.parametrize("kind", ["w2v", "ft"])
def test_results_do_not_depend_on_the_number_of_workers(
    model_name, kind, make_w2v, make_ft, sentences
):
    """With one row per sentence, every worker count gives bit-identical vectors."""
    import fse
    from fse import IndexedList

    wv = make_w2v(DIM) if kind == "w2v" else make_ft(DIM)
    wv.wv.vectors[:] = np.random.default_rng(0).random(wv.wv.vectors.shape, dtype=np.float32)
    if kind == "ft":
        rng = np.random.default_rng(1)
        wv.wv.vectors_vocab[:] = rng.random(wv.wv.vectors_vocab.shape, dtype=np.float32)
        wv.wv.vectors_ngrams[:] = rng.random(wv.wv.vectors_ngrams.shape, dtype=np.float32)
    data = IndexedList([[*s, "unknownword"] for s in sentences] * 20)
    results = []
    for workers in (1, 2, 4, 8):
        model = getattr(fse, model_name)(wv, workers=workers)
        model.train(data)
        results.append(np.array(model.sv.vectors))
    for result in results[1:]:
        assert np.array_equal(result, results[0])


def test_replaced_do_train_job_keeps_the_job_path(w2v, corpus_indexed):
    """An instance that replaces _do_train_job receives lists of tuples, as before."""
    model = Average(w2v)
    seen = []
    original = model._do_train_job

    def spy(data_iterable, target, memory):
        seen.append(type(data_iterable))
        return original(data_iterable=data_iterable, target=target, memory=memory)

    model._do_train_job = spy
    model.train(corpus_indexed)
    assert seen
    assert set(seen) == {list}


def test_encoder_errors_reach_the_caller(w2v, corpus_indexed, monkeypatch):
    """An exception while encoding in the producer thread is raised by train."""
    model = Average(w2v, workers=2)

    def broken(context, sentences):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(model, "_job_encoder", broken)
    with pytest.raises(RuntimeError, match="encoder failed"):
        model.train(corpus_indexed)


def test_kernel_context_cannot_be_pickled(w2v):
    import pickle

    from fse.models.average_inner import KernelContext

    model = Average(w2v)
    _prepare(model, total_sentences=1)
    with pytest.raises(TypeError, match="only exists"):
        pickle.dumps(KernelContext(model, model.sv.vectors))


def _python_batches(items, batch_words):
    """The batching of the job path: words and items both capped at batch_words."""
    batches, batch, size = [], [], 0
    for item in items:
        if size + len(item[0]) <= batch_words and len(batch) < batch_words:
            batch.append(item)
            size += len(item[0])
        else:
            if batch:
                batches.append(batch)
            batch, size = [item], len(item[0])
    if batch:
        batches.append(batch)
    return batches


def _encode_all(model, items, batch_words):
    """Encode every job first, so later calls reuse and grow the buffers, then return them."""
    from fse.models.average_inner import JobEncoder, KernelContext

    context = KernelContext(model, model.sv.vectors)
    encoder = JobEncoder(context, items, batch_words)
    jobs = []
    while (job := encoder.next_job()) is not None:
        jobs.append(job)
    assert encoder.next_job() is None  # end of input stays the end
    return context, jobs


@pytest.mark.parametrize("batch_words", [1, 3, 10, 1000])
@pytest.mark.parametrize("kind", ["w2v", "ft"])
def test_job_encoder_batches_and_encodes_like_the_job_path(w2v, ft, kind, batch_words):
    """Same batches, counts and vectors as the numpy fallback on the Python batches."""
    from fse.models.average_inner import run_job

    rng = np.random.default_rng(0)
    model = Average(w2v if kind == "w2v" else ft)
    words = [*model.wv.index_to_key, "unknownword", "ünbekannt"]
    items = [
        ([words[j] for j in rng.integers(0, len(words), n)], i)
        for i, n in enumerate(rng.choice([0, 0, 1, 2, 5, 12, 40], size=300))
    ]
    _prepare(model, total_sentences=len(items))
    context, jobs = _encode_all(model, items, batch_words)
    batches = _python_batches(items, batch_words)
    assert [job.n_items for job in jobs] == [len(batch) for batch in batches]

    mem = model._get_thread_working_mem()
    counts = [run_job(context, job, mem[0]) for job in jobs]
    encoded = np.array(model.sv.vectors)
    _prepare(model, total_sentences=len(items))
    expected_counts = [train_average_np(model, batch, model.sv.vectors, mem) for batch in batches]
    assert counts == expected_counts
    np.testing.assert_allclose(encoded, model.sv.vectors, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("items", [[], [([], 0), ([], 1), ([], 2)]], ids=["no input", "only empty"])
def test_job_encoder_edge_inputs(w2v, items):
    model = Average(w2v)
    _prepare(model, total_sentences=max(len(items), 1))
    _, jobs = _encode_all(model, items, 2)
    assert [job.n_items for job in jobs] == [len(batch) for batch in _python_batches(items, 2)]
    assert all(job.n_sentences == 0 and job.eff_words == 0 for job in jobs)


def test_job_encoder_keeps_a_huge_reported_length_alone(w2v):
    """A sentence reporting sys.maxsize words is a job of its own, without overflow."""
    import sys

    class Huge(list):
        def __len__(self):
            return sys.maxsize

    model = Average(w2v)
    _prepare(model, total_sentences=3)
    word = model.wv.index_to_key[0]
    _, jobs = _encode_all(model, [([word], 0), (Huge([word]), 1), ([word], 2)], 3)
    assert [job.n_items for job in jobs] == [1, 1, 1]


def test_job_encoder_refuses_to_continue_after_an_error(w2v):
    from fse.models.average_inner import JobEncoder, KernelContext

    model = Average(w2v)
    _prepare(model, total_sentences=2)
    word = model.wv.index_to_key[0]
    encoder = JobEncoder(KernelContext(model, model.sv.vectors), [([word], 0), ([word], 5)], 10)
    with pytest.raises(IndexError):
        encoder.next_job()
    with pytest.raises(RuntimeError, match="failed before"):
        encoder.next_job()


def test_cy_formats_str_subclasses_like_np(ft):
    """An unknown word of a str subclass is hashed from str() of it, as gensim does."""

    class Shouting(str):
        def __str__(self):
            return self.upper()

    model = Average(ft)
    outputs = []
    for train_func in (train_average_np, train_average_cy):
        _prepare(model, total_sentences=1)
        train_func(
            model, [([Shouting("ünbekannt")], 0)], model.sv.vectors, model._get_thread_working_mem()
        )
        outputs.append(np.array(model.sv.vectors))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)


def test_cy_skips_falsy_sentences_like_np(w2v):
    class Falsy(list):
        def __bool__(self):
            return False

    model = Average(w2v)
    words = list(model.wv.index_to_key)[:3]
    outputs = []
    for train_func in (train_average_np, train_average_cy):
        _prepare(model, total_sentences=2)
        counts = train_func(
            model,
            [(Falsy(words), 0), (words, 1)],
            model.sv.vectors,
            model._get_thread_working_mem(),
        )
        outputs.append((counts, np.array(model.sv.vectors)))
    assert outputs[0][0] == outputs[1][0]
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=1e-5, atol=1e-7)


def test_job_encoder_hashes_numpy_strings_like_str(ft):
    """Unknown FastText words given as numpy str_ give the same vectors as str."""
    model = Average(ft)
    outputs = []
    for token in ("ünbekannt", np.str_("ünbekannt")):
        _prepare(model, total_sentences=1)
        train_average_cy(model, [([token], 0)], model.sv.vectors, model._get_thread_working_mem())
        outputs.append(np.array(model.sv.vectors))
    assert np.array_equal(outputs[0], outputs[1])


def test_job_encoder_rejects_unencodable_unknown_words(ft):
    model = Average(ft)
    _prepare(model, total_sentences=1)
    with pytest.raises(UnicodeEncodeError):
        train_average_cy(
            model, [(["\ud800"], 0)], model.sv.vectors, model._get_thread_working_mem()
        )


class _LyingList(list):
    """A list subclass that reports a wrong length."""

    def __len__(self):
        return 1


class _BrokenHash(str):
    def __hash__(self):
        raise RuntimeError("hash failed")


def test_cy_handles_huge_batch_ngrams_like_np(ft):
    """The ngram buffer is sized by the longest unknown word, not by batch_ngrams."""
    model = Average(ft)
    model.batch_ngrams = 2**62 + 1
    sentences = [(["averyveryverylongunknownword", "the"], 0)]
    outputs = []
    for train_func in (train_average_np, train_average_cy):
        _prepare(model, total_sentences=1)
        train_func(model, sentences, model.sv.vectors, model._get_thread_working_mem())
        outputs.append(np.array(model.sv.vectors))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)


def test_cy_does_not_trust_the_reported_sentence_length(w2v):
    """A sentence whose len() is wrong cannot overrun the job buffers."""
    model = Average(w2v)
    words = list(model.wv.index_to_key)[:50]
    outputs = []
    for train_func in (train_average_np, train_average_cy):
        _prepare(model, total_sentences=1)
        train_func(
            model, [(_LyingList(words), 0)], model.sv.vectors, model._get_thread_working_mem()
        )
        outputs.append(np.array(model.sv.vectors))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)


def test_cy_propagates_errors_of_the_vocabulary_lookup(w2v):
    model = Average(w2v)
    _prepare(model, total_sentences=1)
    with pytest.raises(RuntimeError, match="hash failed"):
        train_average_cy(
            model, [([_BrokenHash("the")], 0)], model.sv.vectors, model._get_thread_working_mem()
        )


def test_train_and_infer_each_use_encoded_jobs(w2v, corpus_indexed, monkeypatch):
    model = Average(w2v, workers=2)
    calls = []
    original = Average._run_encoded_job
    monkeypatch.setattr(
        model, "_run_encoded_job", lambda *a: calls.append(1) or original(model, *a)
    )
    model.train(corpus_indexed)
    assert calls
    calls.clear()
    model.infer(corpus_indexed)
    assert calls


@pytest.mark.parametrize("call", ["train", "infer"])
def test_subclass_with_own_do_train_job_keeps_the_job_path(w2v, corpus_indexed, call):
    seen = []

    class Custom(Average):
        def _do_train_job(self, data_iterable, target, memory):
            seen.append(type(data_iterable))
            return super()._do_train_job(data_iterable, target, memory)

    model = Custom(w2v)
    if call == "infer":
        model.train(corpus_indexed)
        seen.clear()
    getattr(model, call)(corpus_indexed)
    assert seen
    assert set(seen) == {list}


def test_encoder_failure_after_queued_jobs_stops_training(w2v, corpus_indexed):
    """The encoder fails after several jobs reached the workers; train raises and ends."""
    import threading

    model = Average(w2v, workers=2)
    model.batch_words = 5

    def failing_input():
        yield from list(corpus_indexed)[:60]
        raise OSError("input failed")

    class Input:
        def __iter__(self):
            return failing_input()

    model.prep.prepare_vectors(sv=model.sv, total_sentences=len(corpus_indexed))
    context = model._kernel_context(model.sv.vectors)
    threads_before = threading.active_count()
    error = _run_with_timeout(
        lambda: model._train_manager(data_iterable=Input(), total_sentences=100, context=context)
    )
    assert isinstance(error, OSError)
    assert str(error) == "input failed"
    assert threading.active_count() <= threads_before


def test_worker_failure_with_encoded_jobs_stops_training(w2v, corpus_indexed, monkeypatch):
    import threading

    model = Average(w2v, workers=3)
    model.batch_words = 5
    done = []

    def failing(context, job, memory):
        done.append(1)
        if len(done) == 3:
            raise RuntimeError("worker failed")
        return job.n_sentences, job.eff_words

    monkeypatch.setattr(model, "_run_encoded_job", failing)
    threads_before = threading.active_count()
    error = _run_with_timeout(lambda: model.train(corpus_indexed))
    assert isinstance(error, RuntimeError)
    assert str(error) == "worker failed"
    assert threading.active_count() <= threads_before


def _run_with_timeout(func, timeout=30):
    """Run func in a thread and return its exception, failing if it does not finish."""
    import threading

    result = {}

    def target():
        try:
            func()
        except BaseException as err:
            result["error"] = err

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "training did not finish; errors must not deadlock"
    return result.get("error")


def test_average_trains_word_vectors_without_counts():
    """Average needs no word frequencies; KeyedVectors built by hand have no counts."""
    from gensim.models import KeyedVectors

    from fse import IndexedList

    kv = KeyedVectors(4)
    kv.add_vectors(["a", "b"], np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    model = Average(kv)
    model.train(IndexedList([["a", "b"], ["b"]]))
    np.testing.assert_allclose(model.sv.vectors, [[0.5, 0.5, 0, 0], [0, 1, 0, 0]], atol=1e-6)


@pytest.mark.parametrize("model_name", ["SIF", "uSIF"])
def test_frequency_models_require_counts(model_name):
    from gensim.models import KeyedVectors

    import fse

    kv = KeyedVectors(4)
    kv.add_vectors(["a", "b"], np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="no 'count' attribute"):
        getattr(fse, model_name)(kv).train(fse.IndexedList([["a", "b"]]))


def test_fasttext_without_ngram_buckets_trains():
    """gensim supports bucket=0; unknown words then add no vector in both code paths."""
    from gensim.models import FastText

    from fse import IndexedList

    ft = FastText(vector_size=4, min_count=1, bucket=0)
    ft.build_vocab([["a", "b"]])
    ft.wv.vectors_vocab[:] = 0
    ft.wv.vectors_vocab[ft.wv.key_to_index["a"], 0] = 1
    ft.wv.vectors_vocab[ft.wv.key_to_index["b"], 1] = 1
    data = IndexedList([["a", "zz"], ["b"]])
    model = Average(ft)
    model.train(data)
    expected = np.array([[0.5, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(model.sv.vectors, expected, atol=1e-6)
    _prepare(model, total_sentences=2)
    train_average_np(model, list(data), model.sv.vectors, model._get_thread_working_mem())
    np.testing.assert_allclose(model.sv.vectors, expected, atol=1e-6)
