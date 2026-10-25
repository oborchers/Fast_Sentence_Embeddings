#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Automated tests for checking the base_s2v class."""

import numpy as np
import pytest
from gensim.models import FastText, Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from wordfreq import get_frequency_dict

from fse.models.average_inner import MAX_WORDS_IN_BATCH, train_average_cy
from fse.models.base_s2v import EPS, BaseSentence2VecModel, BaseSentence2VecPreparer

DIM = 5


@pytest.fixture
def w2v(make_w2v):
    return make_w2v(DIM)


@pytest.fixture
def ft(make_ft):
    return make_ft(DIM)


@pytest.fixture
def corpus_indexed(sentences):
    return [(s, i) for i, s in enumerate(sentences)]


def _count_rows(data_iterable, target, memory):
    """Stand-in for _do_train_job that adds one to the target per sentence."""
    for _i in data_iterable:
        target += 1
    return target


def _pass_method(**kwargs):
    pass


class TestBaseSentence2VecModel:
    def test_log_train_end_with_zero_time(self, w2v):
        """Coarse timers can report zero seconds for a short training run."""
        se = BaseSentence2VecModel(w2v)
        se._log_train_end(eff_sentences=10, eff_words=50, overall_time=0.0)

    def test_init_w_wrong_model(self):
        with pytest.raises(RuntimeError):
            BaseSentence2VecModel(int)

    def test_init_w_empty_w2v_model(self):
        w2v = Word2Vec(min_count=1, vector_size=DIM)
        del w2v.wv.vectors
        with pytest.raises(RuntimeError):
            BaseSentence2VecModel(w2v)

    def test_init_w_empty_vocab_model(self):
        w2v = Word2Vec(min_count=1, vector_size=DIM)
        del w2v.wv
        with pytest.raises(RuntimeError):
            BaseSentence2VecModel(w2v)

    def test_init_w_ft_model_wo_vecs(self, sentences):
        ft = FastText(sentences, vector_size=5)
        ft.wv.vectors_vocab = None
        with pytest.raises(RuntimeError):
            BaseSentence2VecModel(ft)
        ft.wv.vectors_ngrams = None
        with pytest.raises(RuntimeError):
            BaseSentence2VecModel(ft)

    def test_init_w_empty_ft_model(self):
        ft = FastText(min_count=1, vector_size=DIM)
        ft.wv.vectors = np.zeros(10)
        ft.wv.vectors_ngrams = None
        with pytest.raises(RuntimeError):
            BaseSentence2VecModel(ft)

    def test_init_w_incompatible_ft_model(self):
        ft = FastText(min_count=1, vector_size=DIM)
        with pytest.raises(RuntimeError):
            BaseSentence2VecModel(ft)

    def test_include_model(self, w2v):
        se = BaseSentence2VecModel(w2v)
        assert isinstance(se.wv, KeyedVectors)

    def test_model_w_language(self, w2v):
        se = BaseSentence2VecModel(w2v, lang_freq="en")
        freq = int((2**31 - 1) * get_frequency_dict("en", wordlist="best")["help"])
        assert se.wv.get_vecattr("help", "count") == freq
        assert se.wv.get_vecattr("79", "count") == 21

    def test_model_w_wrong_language(self, w2v):
        with pytest.raises(ValueError, match="not available in wordfreq"):
            BaseSentence2VecModel(w2v, lang_freq="test")

    def test_save_load(self, w2v, tmp_path):
        se = BaseSentence2VecModel(w2v)
        p = tmp_path / "test_emb.model"
        se.save(str(p))
        assert p.exists()
        se2 = BaseSentence2VecModel.load(str(p))
        assert (se.wv.vectors == se2.wv.vectors).all()
        assert se.wv.index_to_key == se2.wv.index_to_key
        assert se.workers == se2.workers

    def test_save_load_with_memmap(self, ft, tmp_path):
        shape = (1000, 1000)
        ft.wv.vectors = np.zeros(shape, np.float32)

        p = tmp_path / "test_emb"
        p_vecs = tmp_path / "test_emb_wv.vectors"
        p_ngrams = tmp_path / "test_emb_ngrams.vectors"
        p_vocab = tmp_path / "test_emb_vocab.vectors"

        p_not_exists = tmp_path / "test_emb.wv.vectors.npy"

        se = BaseSentence2VecModel(ft, wv_mapfile_path=str(p))
        assert p_vecs.exists()
        assert p_ngrams.exists()
        assert p_vocab.exists()

        se.save(str(p))
        assert p.exists()
        assert not p_not_exists.exists()

        se = BaseSentence2VecModel.load(str(p))
        assert not se.wv.vectors_vocab.flags.writeable
        assert se.wv.vectors.shape == shape
        assert se.wv.vectors_ngrams.shape == (2000000, 5)

    def test_map_all_vectors_to_disk(self, ft, tmp_path):
        p = tmp_path / "test_emb"

        BaseSentence2VecModel(ft, wv_mapfile_path=str(p))

        assert (tmp_path / "test_emb_wv.vectors").exists()
        assert (tmp_path / "test_emb_ngrams.vectors").exists()
        assert (tmp_path / "test_emb_vocab.vectors").exists()

    @pytest.mark.parametrize(
        "data_iterable",
        [None, "Hello there!", object()],
        ids=["none", "str", "no_iter"],
    )
    def test_input_check(self, w2v, data_iterable):
        se = BaseSentence2VecModel(w2v)
        with pytest.raises(TypeError):
            se._check_input_data_sanity(data_iterable=data_iterable)

    def test_scan_w_list(self, w2v, sentences):
        se = BaseSentence2VecModel(w2v)
        with pytest.raises(TypeError):
            se.scan_sentences(sentences)

    def test_str_rep(self, w2v):
        output = str(BaseSentence2VecModel(w2v))
        assert output == "BaseSentence2VecModel based on KeyedVectors, vector_size=0"

    def test_scan_w_ituple(self, w2v, sentences):
        # The unittest version ran after test_scan_w_empty, which emptied these
        # sentences in the shared corpus; the expected statistics depend on it.
        for i in [5, 10, 15]:
            sentences[i] = []
        se = BaseSentence2VecModel(w2v)
        id_sent = [(s, i) for i, s in enumerate(sentences)]
        stats = se.scan_sentences(id_sent, progress_per=0)

        assert stats["total_sentences"] == 100
        assert stats["total_words"] == 1417
        assert stats["average_length"] == 14
        assert stats["empty_sentences"] == 3
        assert stats["max_index"] == 100

    def test_scan_w_wrong_tuple(self, w2v, sentences):
        se = BaseSentence2VecModel(w2v)
        id_sent = [(s, str(i)) for i, s in enumerate(sentences)]
        with pytest.raises(TypeError):
            se.scan_sentences(id_sent)

    def test_scan_w_empty(self, w2v, sentences):
        se = BaseSentence2VecModel(w2v)
        for i in [5, 10, 15]:
            sentences[i] = []
        output = se.scan_sentences([(s, i) for i, s in enumerate(sentences)])
        assert output["empty_sentences"] == 3

    def test_scan_w_wrong_input(self, w2v, sentences):
        se = BaseSentence2VecModel(w2v)
        raw = ["the dog hit the car", "he was very fast"]

        with pytest.raises(TypeError):
            se.scan_sentences(raw)
        with pytest.raises(TypeError):
            se.scan_sentences([(s, i) for i, s in enumerate(raw)])
        with pytest.raises(TypeError):
            se.scan_sentences([list(range(10) for _ in range(2))])

        with pytest.raises(RuntimeError):
            se.scan_sentences([(s, i + 1) for i, s in enumerate(sentences)])
        with pytest.raises(ValueError, match="Passed negative"):
            se.scan_sentences([(s, i - 1) for i, s in enumerate(sentences)])

    def test_scan_w_many_to_one_input(self, w2v, sentences):
        se = BaseSentence2VecModel(w2v)
        output = se.scan_sentences([(s, 0) for s in sentences])["max_index"]
        assert output == 1

    def test_estimate_memory(self, ft):
        se = BaseSentence2VecModel(ft)
        assert se.estimate_memory(int(1e8))["Total"] == 2040025124

    def test_train(self, w2v, corpus_indexed):
        se = BaseSentence2VecModel(w2v)
        with pytest.raises(NotImplementedError):
            se.train(corpus_indexed)

    def test_log_end(self, w2v):
        se = BaseSentence2VecModel(w2v)
        se._log_train_end(eff_sentences=2000, eff_words=4000, overall_time=10)

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_do_train_job", (None, None, None)),
            ("_pre_train_calls", ()),
            ("_post_train_calls", ()),
            ("_check_parameter_sanity", ()),
            ("_check_dtype_santiy", ()),
            ("_post_inference_calls", ()),
        ],
    )
    def test_child_requirements(self, w2v, method, args):
        se = BaseSentence2VecModel(w2v)
        with pytest.raises(NotImplementedError):
            getattr(se, method)(*args)

    def test_check_pre_train_san_no_wv(self, ft):
        se = BaseSentence2VecModel(ft)
        se.wv = None
        with pytest.raises(RuntimeError):
            se._check_pre_training_sanity(1, 1, 1)

    def test_check_pre_train_san_no_wv_len(self, ft):
        se = BaseSentence2VecModel(ft)
        se.wv.vectors = []
        with pytest.raises(RuntimeError):
            se._check_pre_training_sanity(1, 1, 1)

    def test_check_pre_train_san_no_ngrams_vectors(self, ft):
        se = BaseSentence2VecModel(ft)
        se.wv.vectors_ngrams = []
        with pytest.raises(RuntimeError):
            se._check_pre_training_sanity(1, 1, 1)
        se.wv.vectors_ngrams = [1]
        se.wv.vectors_vocab = []
        with pytest.raises(RuntimeError):
            se._check_pre_training_sanity(1, 1, 1)

    def test_check_pre_train_san_no_sv_vecs(self, ft):
        se = BaseSentence2VecModel(ft)
        se.sv.vectors = None
        with pytest.raises(RuntimeError):
            se._check_pre_training_sanity(1, 1, 1)

    def test_check_pre_train_san_no_word_weights(self, ft):
        se = BaseSentence2VecModel(ft)
        se.word_weights = None
        with pytest.raises(RuntimeError):
            se._check_pre_training_sanity(1, 1, 1)

    def test_check_pre_train_san_incos_len(self, ft):
        se = BaseSentence2VecModel(ft)
        se.word_weights = np.ones(20)
        with pytest.raises(RuntimeError):
            se._check_pre_training_sanity(1, 1, 1)

    def test_check_pre_train_dtypes(self, ft):
        se = BaseSentence2VecModel(ft)

        se.wv.vectors = np.zeros((len(se.wv), 20), dtype=np.float64)
        with pytest.raises(TypeError):
            se._check_pre_training_sanity(1, 1, 1)
        se.wv.vectors = np.zeros((len(se.wv), 20), dtype=np.float32)

        se.wv.vectors_ngrams = np.ones(len(se.wv), dtype=np.float16)
        with pytest.raises(TypeError):
            se._check_pre_training_sanity(1, 1, 1)
        se.wv.vectors_ngrams = np.ones(len(se.wv), dtype=np.float32)

        se.wv.vectors_vocab = np.ones(len(se.wv), dtype=np.float16)
        with pytest.raises(TypeError):
            se._check_pre_training_sanity(1, 1, 1)
        se.wv.vectors_vocab = np.ones(len(se.wv), dtype=np.float32)

        se.sv.vectors = np.zeros((len(se.wv), 20), dtype=int)
        with pytest.raises(TypeError):
            se._check_pre_training_sanity(1, 1, 1)
        se.sv.vectors = np.zeros((len(se.wv), 20), dtype=np.float32)

        se.word_weights = np.ones(len(se.wv), dtype=bool)
        with pytest.raises(TypeError):
            se._check_pre_training_sanity(1, 1, 1)
        se.word_weights = np.ones(len(se.wv), dtype=np.float32)

    def test_check_pre_train_statistics(self, ft):
        se = BaseSentence2VecModel(ft)

        for v in se.wv.key_to_index:
            se.wv.set_vecattr(v, "count", 1)

        # Just throws multiple warnings warning
        se._check_pre_training_sanity(1, 1, 1)

    @pytest.mark.parametrize("args", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
    def test_check_pre_train_statistics_invalid(self, ft, args):
        se = BaseSentence2VecModel(ft)
        for v in se.wv.key_to_index:
            se.wv.set_vecattr(v, "count", 1)

        with pytest.raises(ValueError, match="invalid values"):
            se._check_pre_training_sanity(*args)

    @pytest.mark.parametrize("args", [(0, 1), (1, 0)])
    def test_post_training_sanity(self, sentences, args):
        w2v = Word2Vec()
        w2v.build_vocab(sentences)
        se = BaseSentence2VecModel(w2v)
        se.prep.prepare_vectors(se.sv, 20)
        with pytest.raises(ValueError, match="invalid values"):
            se._check_post_training_sanity(*args)

    def test_move_ndarray_to_disk_w2v(self, w2v, tmp_path):
        se = BaseSentence2VecModel(w2v)
        p = tmp_path / "test_vecs"
        p_target = tmp_path / "test_vecs_wv.vectors"
        se.wv.vectors[0, 1] = 10
        vecs = se.wv.vectors.copy()
        output = se._move_ndarray_to_disk(se.wv.vectors, name="wv", mapfile_path=str(p))
        assert p_target.exists()
        assert not output.flags.writeable
        assert (vecs == output).all()

    def test_move_w2v_vectors_to_disk_from_init(self, w2v, tmp_path):
        p = tmp_path / "test_vecs"
        se = BaseSentence2VecModel(w2v, wv_mapfile_path=str(p))
        assert (tmp_path / "test_vecs_wv.vectors").exists()
        assert not se.wv.vectors.flags.writeable

    def test_move_ft_vectors_to_disk_from_init(self, ft, tmp_path):
        p = tmp_path / "test_vecs"

        se = BaseSentence2VecModel(ft, wv_mapfile_path=str(p))

        assert (tmp_path / "test_vecs_wv.vectors").exists()
        assert not se.wv.vectors.flags.writeable

        assert (tmp_path / "test_vecs_ngrams.vectors").exists()
        assert not se.wv.vectors_ngrams.flags.writeable

        assert (tmp_path / "test_vecs_vocab.vectors").exists()

    def test_train_manager(self, w2v, corpus_indexed):
        # batch_words=100 splits the 1450 words of the corpus into 16 jobs,
        # so both workers take jobs from the queue.
        se = BaseSentence2VecModel(w2v, workers=2, batch_words=100)
        job_sizes = []

        def temp_train_job(data_iterable, target, memory):
            v1 = v2 = sum(1 for _ in data_iterable)
            job_sizes.append(v1)
            return v1 * 2, v2 * 3

        se._do_train_job = temp_train_job
        job_output = se._train_manager(
            data_iterable=corpus_indexed,
            total_sentences=len(corpus_indexed),
            report_delay=0.01,
        )
        assert job_output == (100, 200, 300)
        assert len(job_sizes) == 16
        assert sum(job_sizes) == 100

    def test_infer_method(self, w2v, corpus_indexed):
        se = BaseSentence2VecModel(w2v)
        se._post_inference_calls = _pass_method
        se._do_train_job = _count_rows
        output = se.infer(corpus_indexed)
        assert (output == 100).all()

    def test_infer_method_cy_overflow(self, w2v, sentences):
        se = BaseSentence2VecModel(w2v)

        def _do_train_job(data_iterable, target, memory):
            eff_sentences, eff_words = train_average_cy(
                model=se, indexed_sentences=data_iterable, target=target, memory=memory
            )
            return eff_sentences, eff_words

        se._post_inference_calls = _pass_method
        se._do_train_job = _do_train_job
        tmp = []
        for _ in range(20):
            tmp.extend(sentences)
        bs = 0
        last = 0
        for last, s in enumerate(tmp):  # noqa: B007
            if bs >= MAX_WORDS_IN_BATCH:
                break
            bs += len(s)
        sents = [(s, i) for i, s in enumerate(tmp)]
        output = se.infer(sents)
        output = output[last:]
        assert (output != 0).all()

    def test_infer_many_to_one(self, w2v, sentences):
        se = BaseSentence2VecModel(w2v)
        se._post_inference_calls = _pass_method
        se._do_train_job = _count_rows
        output = se.infer([(s, 0) for s in sentences])
        assert (output == 100).all()
        assert output.shape == (1, 5)

    def test_infer_use_norm(self, w2v, corpus_indexed):
        se = BaseSentence2VecModel(w2v)
        se._post_inference_calls = _pass_method
        se._do_train_job = _count_rows
        output = se.infer(corpus_indexed, use_norm=True)

        assert np.allclose(1.0, np.sqrt(np.sum(output[0] ** 2)))


class TestBaseSentence2VecPreparer:
    @pytest.fixture
    def trainables(self):
        return BaseSentence2VecPreparer()

    @pytest.fixture(params=[False, True], ids=["memory", "memmap"])
    def se(self, request, w2v, tmp_path):
        """Model with sentence vectors in memory or mapped to a file in tmp_path."""
        if request.param:
            return BaseSentence2VecModel(w2v, sv_mapfile_path=str(tmp_path / "test_vectors"))
        return BaseSentence2VecModel(w2v)

    @staticmethod
    def _assert_mapfile(se, tmp_path):
        if se.sv.mapfile_path is not None:
            assert (tmp_path / "test_vectors.vectors").exists()

    def test_reset_vectors(self, se, trainables, tmp_path):
        trainables.reset_vectors(se.sv, 20)
        self._assert_mapfile(se, tmp_path)
        assert se.sv.vectors.shape == (20, DIM)
        assert se.sv.vectors.dtype == np.float32
        assert (se.sv.vectors == EPS).all()
        assert se.sv.vectors_norm is None

    def test_update_vectors(self, se, trainables, tmp_path):
        trainables.reset_vectors(se.sv, 20)
        se.sv.vectors[:] = 1.0
        trainables.update_vectors(se.sv, 10)
        self._assert_mapfile(se, tmp_path)
        assert se.sv.vectors.shape == (30, DIM)
        assert se.sv.vectors.dtype == np.float32
        assert (np.ones((20, DIM)) == se.sv.vectors[:20]).all()
        assert (se.sv.vectors[20:] == EPS).all()
        assert se.sv.vectors_norm is None

    def test_prepare_vectors(self, w2v, trainables):
        se = BaseSentence2VecModel(w2v)
        trainables.prepare_vectors(se.sv, 20, update=False)
        assert se.sv.vectors.shape == (20, DIM)
        trainables.prepare_vectors(se.sv, 40, update=True)
        assert se.sv.vectors.shape == (60, DIM)
