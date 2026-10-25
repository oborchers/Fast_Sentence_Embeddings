#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Automated tests for checking the base_s2v class."""

import threading

import numpy as np
import pytest
from gensim.models import FastText, Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from pydantic import ValidationError
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
        with pytest.raises(ValidationError, match="instance of KeyedVectors"):
            BaseSentence2VecModel(int)

    @pytest.mark.parametrize(
        "kwargs",
        [{"workers": 0}, {"workers": "2"}, {"batch_words": 0}, {"sv_mapfile_path": 3}],
    )
    def test_init_rejects_invalid_arguments(self, w2v, kwargs):
        with pytest.raises(ValidationError):
            BaseSentence2VecModel(w2v, **kwargs)

    @pytest.mark.parametrize("kwargs", [{"cache_size_gb": float("inf")}, {"alpha": float("nan")}])
    def test_init_rejects_non_finite_floats(self, w2v, kwargs):
        from fse.models.sif import SIF

        with pytest.raises(ValidationError):
            SIF(w2v, **kwargs)

    def test_init_accepts_numpy_integers(self, w2v):
        from fse.models.sif import SIF

        se = SIF(w2v, components=np.int64(2), workers=np.int32(2))
        assert se.components == 2
        assert type(se.components) is int
        with pytest.raises(ValidationError):
            SIF(w2v, workers=np.bool_(True))

    def test_init_accepts_path_objects(self, w2v, tmp_path):
        se = BaseSentence2VecModel(w2v, sv_mapfile_path=tmp_path / "sv")
        assert se.sv.mapfile_path == tmp_path / "sv"

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
        vectors_vocab = ft.wv.vectors_vocab
        ft.wv.vectors_vocab = None
        with pytest.raises(RuntimeError, match="vectors_vocab"):
            BaseSentence2VecModel(ft)
        ft.wv.vectors_vocab = vectors_vocab
        ft.wv.vectors_ngrams = None
        with pytest.raises(RuntimeError, match="Ngram vectors"):
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

    def test_save_with_memmap_keeps_live_model(self, ft, tmp_path):
        """Saving must not remove the vectors from the model that is still in use."""
        from fse.models.average import Average

        se = Average(ft, wv_mapfile_path=str(tmp_path / "emb"))
        vectors = se.wv.vectors
        expected = se.infer([(["the", "product"], 0)])
        se.save(str(tmp_path / "model"))
        assert se.wv.vectors is vectors
        assert np.array_equal(se.infer([(["the", "product"], 0)]), expected)

    def test_save_with_memmap_skips_mapped_vectors(self, ft, tmp_path):
        """wv.vectors lives in the mapfile and is not written a second time as .npy file."""
        se = BaseSentence2VecModel(ft, wv_mapfile_path=str(tmp_path / "emb"))
        # gensim keeps small arrays inside the pickle; sep_limit=0 stores every array separately.
        se.save(str(tmp_path / "model"), sep_limit=0)
        assert (tmp_path / "model.wv.vectors_vocab.npy").exists()
        assert not (tmp_path / "model.wv.vectors.npy").exists()

    def test_save_load_with_memmap_twice(self, ft, tmp_path):
        se = BaseSentence2VecModel(ft, wv_mapfile_path=str(tmp_path / "emb"))
        expected = np.array(se.wv.vectors_vocab)
        for cycle in range(2):
            path = str(tmp_path / f"model_{cycle}")
            se.save(path)
            se = BaseSentence2VecModel.load(path)
            assert np.array_equal(se.wv.vectors_vocab, expected)

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
        assert output == "BaseSentence2VecModel based on KeyedVectors, vector_size=5"

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

    @pytest.mark.parametrize(
        ("bad_item", "error", "match"),
        [
            ((["a", "b"], -1), ValueError, "At 10: Passed negative"),
            ((["a", "b"], "3"), TypeError, "At 10: .*tuple.index"),
            ((["a", "b"], 2.0), TypeError, "At 10: .*tuple.index"),
            (("a b", 3), TypeError, "At 10: .*tuple.words"),
            ((["a", 2], 3), TypeError, "At 10: .*tuple.words"),
        ],
    )
    def test_scan_validates_every_item(self, w2v, sentences, bad_item, error, match):
        """Invalid items are rejected wherever they appear, not only in first position."""
        se = BaseSentence2VecModel(w2v)
        items = [(s, i) for i, s in enumerate(sentences)]
        items[10] = bad_item
        with pytest.raises(error, match=match):
            se.scan_sentences(items)

    def test_scan_accepts_numpy_integer_index(self, w2v, sentences):
        se = BaseSentence2VecModel(w2v)
        items = [(s, np.int64(i)) for i, s in enumerate(sentences)]
        assert se.scan_sentences(items)["max_index"] == len(sentences)

    @pytest.mark.parametrize("method", ["train", "infer"])
    def test_rejects_one_shot_iterators(self, w2v, sentences, method):
        """A generator would be exhausted by the scan and then train on nothing."""
        se = BaseSentence2VecModel(w2v)
        generator = ((s, i) for i, s in enumerate(sentences))
        with pytest.raises(TypeError, match="re-iterable"):
            getattr(se, method)(generator)

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

    @pytest.mark.parametrize(
        ("batch_words", "expected_jobs"),
        [
            # The default batch holds the whole corpus of 1450 words in one job.
            (10_000, 1),
            # batch_words=100 splits the corpus into 16 jobs across both workers.
            (100, 16),
        ],
    )
    def test_train_manager(self, w2v, corpus_indexed, batch_words, expected_jobs):
        se = BaseSentence2VecModel(w2v, workers=2, batch_words=batch_words)
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
        assert len(job_sizes) == expected_jobs
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

    def test_fill_crosses_block_boundaries(self, se, trainables, monkeypatch):
        """Rows are filled in blocks; sizes that are no multiple of the block are covered."""
        from fse.models import base_s2v

        monkeypatch.setattr(base_s2v, "_FILL_BLOCK_ROWS", 7)
        trainables.reset_vectors(se.sv, 20)
        se.sv.vectors[:] = 1.0
        trainables.update_vectors(se.sv, 10)
        assert (se.sv.vectors[:20] == 1.0).all()
        assert (se.sv.vectors[20:] == EPS).all()

    def test_prepare_vectors(self, w2v, trainables):
        se = BaseSentence2VecModel(w2v)
        trainables.prepare_vectors(se.sv, 20, update=False)
        assert se.sv.vectors.shape == (20, DIM)
        trainables.prepare_vectors(se.sv, 40, update=True)
        assert se.sv.vectors.shape == (60, DIM)


def _run_with_timeout(func, timeout=30):
    """Run func in a thread and return its exception, failing if it does not finish."""
    result = {}

    def target():
        try:
            func()
        except BaseException as err:
            result["error"] = err

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "training did not finish; worker errors must not deadlock"
    return result.get("error")


class TestTrainManagerErrors:
    @pytest.mark.parametrize("workers", [1, 4])
    def test_worker_exception_propagates(self, w2v, corpus_indexed, workers):
        se = BaseSentence2VecModel(w2v, workers=workers, batch_words=100)
        calls = []

        def failing_job(data_iterable, target, memory):
            calls.append(1)
            if len(calls) == 3:
                raise ValueError("job failed")
            return len(data_iterable), len(data_iterable)

        se._do_train_job = failing_job
        threads_before = threading.active_count()
        error = _run_with_timeout(
            lambda: se._train_manager(
                data_iterable=corpus_indexed, total_sentences=len(corpus_indexed)
            )
        )
        assert isinstance(error, ValueError)
        assert str(error) == "job failed"
        assert threading.active_count() <= threads_before

    def test_interrupted_progress_does_not_leave_threads(self, w2v, corpus_indexed):
        """An exception in the main thread, such as Ctrl+C, must not leave blocked workers."""
        se = BaseSentence2VecModel(w2v, workers=2, batch_words=5)
        se._do_train_job = lambda data_iterable, target, memory: (len(data_iterable), 1)

        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        se._log_train_progress = interrupted
        threads_before = threading.active_count()
        error = _run_with_timeout(
            lambda: se._train_manager(
                data_iterable=corpus_indexed, total_sentences=len(corpus_indexed)
            )
        )
        assert isinstance(error, KeyboardInterrupt)
        assert threading.active_count() <= threads_before

    def test_input_exception_propagates(self, w2v, corpus_indexed):
        se = BaseSentence2VecModel(w2v, workers=2, batch_words=100)
        se._do_train_job = lambda data_iterable, target, memory: (len(data_iterable), 0)

        def broken_input():
            yield from list(corpus_indexed)[:50]
            raise OSError("input failed")

        error = _run_with_timeout(
            lambda: se._train_manager(data_iterable=broken_input(), total_sentences=100)
        )
        assert isinstance(error, OSError)
        assert str(error) == "input failed"


class TestSmallFixes:
    def test_str_shows_vector_size(self, w2v):
        se = BaseSentence2VecModel(w2v)
        assert f"vector_size={w2v.wv.vector_size}" in str(se)

    def test_scan_short_sentences(self, w2v):
        """An average below one word per sentence is valid input."""
        se = BaseSentence2VecModel(w2v)
        stats = se.scan_sentences([([], 0), ([], 1), (["the"], 2)])
        assert stats["total_words"] == 1
        assert stats["average_length"] >= 1

    def test_scan_empty_input(self, w2v):
        se = BaseSentence2VecModel(w2v)
        stats = se.scan_sentences([])
        assert stats["total_sentences"] == 0
        with pytest.raises(ValueError, match="invalid values"):
            se._check_pre_training_sanity(**stats)

    def test_unknown_keyword_arguments_raise(self, w2v):
        """A typo such as worker=4 must not be ignored silently."""
        from fse.models.average import Average

        with pytest.raises(ValidationError, match="Unexpected keyword argument"):
            Average(w2v, foo=1)
        with pytest.raises(ValidationError, match="Unexpected keyword argument"):
            BaseSentence2VecModel(w2v, worker=4)

    def test_tuple_with_extra_element_raises(self, w2v):
        se = BaseSentence2VecModel(w2v)
        with pytest.raises(ValueError, match="tuple of length 3"):
            se.scan_sentences([(["the"], 0, "ignored")])


class TestUniqueIndices:
    @pytest.mark.parametrize(
        ("indices", "unique"),
        [
            ([0, 1, 2, 3], True),
            ([3, 1, 0, 2], True),
            ([0, 1, 0, 2], False),
            ([0, 0, 2], False),
            ([1, 1, 1, 1], False),
        ],
    )
    def test_scan_detects_duplicate_indices(self, w2v, indices, unique):
        se = BaseSentence2VecModel(w2v)
        items = [(["the"], index) for index in indices]
        assert se.scan_sentences(items)["unique_indices"] is unique


def test_many_to_one_without_extension_uses_one_worker(w2v, sentences, monkeypatch, caplog):
    """Without the C extension there are no row locks, so training falls back to one worker."""
    import logging

    from fse.models import base_s2v
    from fse.models.average import Average

    monkeypatch.setattr(base_s2v, "RowLocks", None)
    model = Average(w2v, workers=4)
    used = []
    original = model._train_manager

    def recording_manager(*args, **kwargs):
        used.append(kwargs["workers"])
        return original(*args, **kwargs)

    model._train_manager = recording_manager
    with caplog.at_level(logging.WARNING, logger="fse.models.base_s2v"):
        model.train([(s, i % 3) for i, s in enumerate(sentences)])
    assert used == [1]
    assert "one worker" in caplog.text


def test_row_locks_are_removed_after_training(w2v, sentences, tmp_path):
    from fse.models.average import Average

    model = Average(w2v, workers=2)
    model.train([(s, i % 3) for i, s in enumerate(sentences)])
    assert "_row_locks" not in vars(model)
    model.save(str(tmp_path / "model"))


@pytest.mark.parametrize("lang_freq", [None, "en"])
def test_word_probabilities_match_the_per_word_loop(w2v, lang_freq):
    """The vectorized probabilities are bit-identical to summing get_vecattr per word."""
    se = BaseSentence2VecModel(w2v, lang_freq=lang_freq)
    expected = np.zeros(len(se.wv), dtype=np.float32)
    corpus_size = 0
    for word, index in se.wv.key_to_index.items():
        count = se.wv.get_vecattr(word, "count")
        corpus_size += count
        expected[index] = count
    expected /= corpus_size
    assert np.array_equal(se._word_probabilities(), expected)


def test_word_probabilities_reject_negative_counts(w2v):
    se = BaseSentence2VecModel(w2v)
    se.wv.set_vecattr(se.wv.index_to_key[0], "count", -1)
    with pytest.raises(ValueError, match="negative"):
        se._word_probabilities()


def test_job_producer_caps_jobs_by_sentence_count(w2v):
    """Empty sentences add no words, so jobs are also capped at batch_words sentences."""
    import threading
    from queue import Queue

    se = BaseSentence2VecModel(w2v, batch_words=10)
    queue: Queue = Queue()
    errors: list = []
    se._job_producer([([], i) for i in range(25)], queue, errors, threading.Event(), workers=1)
    jobs = []
    while (job := queue.get()) is not None:
        jobs.append(job)
    assert not errors
    assert [len(job) for job in jobs] == [10, 10, 5]


SCAN_INPUTS = [
    [],
    [(["a", "b"], 0), ([], 1), (["c"], 2)],
    [(["a"], 2), (["b"], 0), (["c"], 1)],
    [(["a"], 0), (["b"], 0), (["c"], 1)],
    [(["a"], np.int64(1)), (["b"], np.int32(0))],
    [(["a"], 0), (["b"], 5)],
    [(["a"], 0), ["b", 1]],
    [(["a"], 0), (["b"], 1, "x")],
    [(["a"], 0), ("b", 1)],
    [(["a"], 0), ([1], 1)],
    [(["a"], 0), (["b"], True)],
    [(["a"], 0), (["b"], 1.0)],
    [(["a"], 0), (["b"], -1)],
    [(["a"], 0), (["b"], np.int64(-3))],
    [(["a", np.str_("b")], 0)],
]


@pytest.mark.parametrize("data", SCAN_INPUTS)
def test_compiled_scan_matches_the_python_loop(w2v, data, monkeypatch):
    """Same statistics, or the same exception type and message, in both scan paths."""
    from fse.models import base_s2v

    def outcome():
        se = BaseSentence2VecModel(w2v)
        try:
            return se.scan_sentences(data)
        except Exception as err:
            return type(err), str(err)

    compiled = outcome()
    monkeypatch.setattr(base_s2v, "scan_indexed_sentences", None)
    assert outcome() == compiled


class _IterOverridingList(list):
    """Stores strings but iterates integers; the Python check iterates."""

    def __iter__(self):
        return iter([1, 2])


def test_compiled_scan_iterates_list_subclasses(w2v, monkeypatch):
    from fse.models import base_s2v

    data = [(_IterOverridingList(["a", "b"]), 0)]

    def outcome():
        try:
            return BaseSentence2VecModel(w2v).scan_sentences(data)
        except Exception as err:
            return type(err), str(err)

    compiled = outcome()
    assert compiled[0] is TypeError
    monkeypatch.setattr(base_s2v, "scan_indexed_sentences", None)
    assert outcome() == compiled


def test_scan_uses_a_validator_replaced_on_the_instance(w2v):
    se = BaseSentence2VecModel(w2v)

    def reject(iter_pos, obj):
        raise ValueError("custom validator")

    se._check_indexed_sent_valid = reject
    with pytest.raises(ValueError, match="custom validator"):
        se.scan_sentences([(["a"], 0)])
