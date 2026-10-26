#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""float16 and int8 word vectors (wv_dtype) and the row kernels that read them."""

import gc

import numpy as np
import pytest
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors
from pydantic import ValidationError

from fse.inputs import IndexedList
from fse.models import average_inner
from fse.models._quantize import STORAGE_TYPES, dequantize, quantize_into
from fse.models.average import Average, train_average_np
from fse.models.average_inner import train_average_cy
from fse.models.base_s2v import BaseSentence2VecModel
from fse.models.sif import SIF
from fse.models.usif import uSIF

# Every test also checks that no file is moved while a memmap still maps it (Windows).
pytestmark = pytest.mark.usefixtures("windows_file_maps")

FORMATS = ["float32", "float16", "int8"]
QUANTIZED = ["float16", "int8"]
# Dimensions around the 8-wide vector loops, so every kernel also runs its tail.
DIMS = [1, 7, 8, 9, 100, 300]


@pytest.fixture(params=[False, True], ids=["fastest", "portable"])
def kernels(request):
    """Run a test with the kernels the CPU supports, and again with the portable fallback."""
    average_inner._use_portable_kernels(request.param)
    yield request.param
    average_inner._use_portable_kernels(False)


def _keyed_vectors(vocab, dim, seed=0):
    rng = np.random.default_rng(seed)
    kv = KeyedVectors(dim)
    kv.add_vectors(
        [f"w{i}" for i in range(vocab)], rng.standard_normal((vocab, dim), dtype=np.float32)
    )
    for i in range(vocab):
        kv.set_vecattr(f"w{i}", "count", vocab - i)
    return kv


def _sentences(vocab, n, seed=1, max_len=30):
    rng = np.random.default_rng(seed)
    return [[f"w{j}" for j in rng.integers(0, vocab, rng.integers(1, max_len))] for _ in range(n)]


def _codes(rows, fmt):
    codes = np.empty(rows.shape, dtype=STORAGE_TYPES[fmt])
    scales = np.empty(len(rows), dtype=np.float32) if fmt == "int8" else None
    quantize_into(rows, codes, scales)
    return codes, scales


def _reference(model, sentences):
    """float64 averages of the stored rows, as the kernel should compute them."""
    words, scales, _, _ = model._word_arrays()
    rows = dequantize(words, scales).astype(np.float64)
    weights = model.word_weights.astype(np.float64)
    out = np.zeros((len(sentences), model.sv.vector_size))
    for i, sentence in enumerate(sentences):
        index = [model.wv.key_to_index[w] for w in sentence]
        out[i] = (weights[index, None] * rows[index]).sum(axis=0) / len(index)
    return out


def _prepare(model, n):
    model.prep.prepare_vectors(sv=model.sv, total_sentences=n, update=False)
    model._pre_train_calls()
    model.sv.vectors[:] = 0
    return model


class TestKernels:
    def test_kernel_info_names_an_instruction_set(self):
        assert average_inner._kernel_info() in {"avx2", "neon", "portable"}

    def test_portable_switch_changes_and_restores_the_kernels(self):
        chosen = average_inner._kernel_info()
        average_inner._use_portable_kernels(True)
        try:
            assert average_inner._kernel_info() == "portable"
        finally:
            average_inner._use_portable_kernels(False)
        assert average_inner._kernel_info() == chosen

    @pytest.mark.parametrize("dim", DIMS)
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_kernel_averages_the_stored_rows(self, kernels, fmt, dim):
        vocab = 257
        kv = _keyed_vectors(vocab, dim)
        sentences = _sentences(vocab, 60)
        model = _prepare(Average(kv, wv_dtype=fmt), len(sentences))
        model.word_weights = np.random.default_rng(2).random(vocab, dtype=np.float32)
        mem = model._get_thread_working_mem()
        train_average_cy(model, [(s, i) for i, s in enumerate(sentences)], model.sv.vectors, mem)
        expected = _reference(model, sentences)
        np.testing.assert_allclose(model.sv.vectors, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_prefetch_at_the_ends_of_sentences_and_jobs(self, kernels, fmt):
        """Sentences shorter and longer than the prefetch distance, the last one ending a job."""
        vocab, dim = 50, 9
        kv = _keyed_vectors(vocab, dim)
        sentences = [[f"w{i % vocab}" for i in range(n)] for n in (1, 7, 8, 9, 17, 64)]
        model = _prepare(Average(kv, wv_dtype=fmt), len(sentences))
        mem = model._get_thread_working_mem()
        train_average_cy(model, [(s, i) for i, s in enumerate(sentences)], model.sv.vectors, mem)
        np.testing.assert_allclose(
            model.sv.vectors, _reference(model, sentences), rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_fasttext_unknown_words_read_the_ngram_codes(self, kernels, make_ft, fmt):
        ft = make_ft(13, bucket=500)
        rng = np.random.default_rng(3)
        ft.wv.vectors_vocab[:] = rng.standard_normal(ft.wv.vectors_vocab.shape).astype(np.float32)
        ft.wv.vectors_ngrams[:] = rng.standard_normal(ft.wv.vectors_ngrams.shape).astype(np.float32)
        sentences = [["unknownword", "anotherunknown"], ["unknownword"]]
        cy = _prepare(Average(ft, wv_dtype=fmt), len(sentences))
        np_model = _prepare(Average(ft, wv_dtype=fmt), len(sentences))
        indexed = [(s, i) for i, s in enumerate(sentences)]
        train_average_cy(cy, indexed, cy.sv.vectors, cy._get_thread_working_mem())
        train_average_np(np_model, indexed, np_model.sv.vectors, np_model._get_thread_working_mem())
        assert np.abs(cy.sv.vectors).max() > 0
        np.testing.assert_allclose(cy.sv.vectors, np_model.sv.vectors, rtol=1e-5, atol=1e-6)

    def test_codes_of_the_wrong_dtype_are_rejected(self, w2v_model):
        model = _prepare(Average(w2v_model, wv_dtype="int8"), 1)
        model.wv_codes = model.wv_codes.astype(np.float16)
        with pytest.raises(TypeError, match="wv_codes must have dtype int8"):
            train_average_cy(
                model, [(["go"], 0)], model.sv.vectors, model._get_thread_working_mem()
            )

    def test_int8_codes_without_scales_are_rejected(self, w2v_model):
        model = _prepare(Average(w2v_model, wv_dtype="int8"), 1)
        model.wv_scales = None
        with pytest.raises(TypeError, match="wv_scales must be a numpy array"):
            train_average_cy(
                model, [(["go"], 0)], model.sv.vectors, model._get_thread_working_mem()
            )


@pytest.fixture
def w2v_model(make_w2v):
    return make_w2v(20)


class TestModel:
    @pytest.mark.parametrize("fmt", QUANTIZED)
    def test_codes_are_the_quantized_word_vectors(self, w2v_model, fmt):
        model = Average(w2v_model, wv_dtype=fmt)
        codes, scales = _codes(w2v_model.wv.vectors, fmt)
        assert model.wv_dtype == fmt
        np.testing.assert_array_equal(model.wv_codes, codes)
        if fmt == "int8":
            np.testing.assert_array_equal(model.wv_scales, scales)
        else:
            assert model.wv_scales is None

    def test_float32_keeps_no_codes(self, w2v_model):
        model = Average(w2v_model)
        assert model.wv_dtype == "float32"
        assert model.wv_codes is None
        assert model._word_arrays()[0] is model.wv.vectors

    @pytest.mark.parametrize("fmt", QUANTIZED)
    def test_word_vectors_passed_in_are_not_changed(self, w2v_model, fmt):
        before = w2v_model.wv.vectors.copy()
        model = Average(w2v_model, wv_dtype=fmt)
        model.train(IndexedList([["go", "pull"]]))
        assert model.wv is w2v_model.wv
        np.testing.assert_array_equal(w2v_model.wv.vectors, before)
        assert w2v_model.wv.vectors.dtype == np.float32

    def test_unknown_format_is_rejected(self, w2v_model):
        with pytest.raises(ValidationError):
            Average(w2v_model, wv_dtype="bfloat16")

    def test_float16_overflow_names_the_word_vector(self, w2v_model):
        w2v_model.wv.vectors[3, 0] = 1e6
        with pytest.raises(
            ValueError, match=r"word vector 3 has values beyond the float16 range.*wv_dtype"
        ):
            Average(w2v_model, wv_dtype="float16")

    def test_non_finite_word_vector_is_rejected(self, w2v_model):
        w2v_model.wv.vectors[5, 1] = np.nan
        with pytest.raises(ValueError, match="word vector 5 contains NaN"):
            Average(w2v_model, wv_dtype="int8")

    @pytest.mark.parametrize("cls", [Average, SIF, uSIF])
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_numpy_fallback_matches_the_kernel(self, w2v_model, sentences, cls, fmt):
        indexed = [(s, i) for i, s in enumerate(sentences)]
        # uSIF needs a sentence length, which train would take from its scan.
        extra = {"length": 11} if cls is uSIF else {}
        cy = _prepare(cls(w2v_model, wv_dtype=fmt, **extra), len(indexed))
        np_model = _prepare(cls(w2v_model, wv_dtype=fmt, **extra), len(indexed))
        assert train_average_cy(cy, indexed, cy.sv.vectors, cy._get_thread_working_mem()) == (
            train_average_np(
                np_model, indexed, np_model.sv.vectors, np_model._get_thread_working_mem()
            )
        )
        np.testing.assert_allclose(cy.sv.vectors, np_model.sv.vectors, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("cls", [Average, SIF, uSIF])
    @pytest.mark.parametrize("fmt", QUANTIZED)
    def test_quantized_words_stay_close_to_float32(self, w2v_model, sentences, cls, fmt):
        w2v_model.wv.vectors[:] = np.random.default_rng(4).standard_normal(
            w2v_model.wv.vectors.shape
        )
        full = cls(w2v_model)
        small = cls(w2v_model, wv_dtype=fmt)
        full.train(IndexedList(sentences))
        small.train(IndexedList(sentences))
        a, b = full.sv.get_vectors(), small.sv.get_vectors()
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        assert cos.min() > (0.9999 if fmt == "float16" else 0.999)

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("sv_dtype", FORMATS)
    def test_results_do_not_depend_on_the_workers(self, w2v_model, sentences, fmt, sv_dtype):
        one = Average(w2v_model, wv_dtype=fmt, sv_dtype=sv_dtype, workers=1)
        four = Average(w2v_model, wv_dtype=fmt, sv_dtype=sv_dtype, workers=4)
        one.train(IndexedList(sentences))
        four.train(IndexedList(sentences))
        np.testing.assert_array_equal(one.sv.vectors, four.sv.vectors)

    @pytest.mark.parametrize("fmt", QUANTIZED)
    def test_infer_and_update_read_the_codes(self, w2v_model, sentences, fmt):
        model = SIF(w2v_model, wv_dtype=fmt)
        model.train(IndexedList(sentences))
        inferred = model.infer(IndexedList(sentences[:5]))
        np.testing.assert_allclose(inferred, model.sv.get_vectors()[:5], rtol=1e-5, atol=1e-6)
        model.train(IndexedList(sentences[:5]), update=True)
        np.testing.assert_allclose(model.sv.get_vectors()[-5:], inferred, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("fmt", QUANTIZED)
    def test_fasttext_model_quantizes_vocab_and_ngrams(self, make_ft, fmt):
        ft: FastText = make_ft(12, bucket=300)
        model = Average(ft, wv_dtype=fmt)
        np.testing.assert_array_equal(model.wv_codes, _codes(ft.wv.vectors_vocab, fmt)[0])
        np.testing.assert_array_equal(model.ngram_codes, _codes(ft.wv.vectors_ngrams, fmt)[0])
        assert (model.ngram_scales is not None) == (fmt == "int8")

    @pytest.mark.parametrize("fmt", QUANTIZED)
    def test_save_and_load_keep_the_codes(self, w2v_model, sentences, tmp_path, fmt):
        model = SIF(w2v_model, wv_dtype=fmt)
        model.train(IndexedList(sentences))
        path = tmp_path / "model"
        model.save(str(path))
        loaded = SIF.load(str(path))
        assert loaded.wv_dtype == fmt
        np.testing.assert_array_equal(loaded.wv_codes, model.wv_codes)
        np.testing.assert_array_equal(
            loaded.infer(IndexedList(sentences[:3])), model.infer(IndexedList(sentences[:3]))
        )

    @pytest.mark.parametrize("fmt", QUANTIZED)
    def test_mapfiles_hold_the_codes(self, w2v_model, sentences, tmp_path, fmt):
        mapfile = tmp_path / "wv"
        model = Average(w2v_model, wv_dtype=fmt, wv_mapfile_path=str(mapfile))
        assert isinstance(model.wv_codes, np.memmap)
        assert (tmp_path / f"wv_wv.{fmt}").exists()
        assert not list(tmp_path.glob("*.writing"))
        model.train(IndexedList(sentences))
        path = tmp_path / "model"
        model.save(str(path))
        loaded = Average.load(str(path))
        assert isinstance(loaded.wv_codes, np.memmap)
        np.testing.assert_array_equal(loaded.wv_codes, model.wv_codes)
        np.testing.assert_array_equal(
            loaded.infer(IndexedList(sentences[:3])), model.infer(IndexedList(sentences[:3]))
        )
        # A second model over the same word vectors reuses the file.
        again = Average(w2v_model, wv_dtype=fmt, wv_mapfile_path=str(mapfile))
        np.testing.assert_array_equal(again.wv_codes, model.wv_codes)

    def test_mapfile_with_other_codes_is_rejected(self, w2v_model, tmp_path):
        mapfile = tmp_path / "wv"
        first = Average(w2v_model, wv_dtype="int8", wv_mapfile_path=str(mapfile))
        del first
        gc.collect()
        # Only the codes file stays, so the check of the codes, not of the float32 copy, fires.
        (tmp_path / "wv_wv.vectors").unlink()
        w2v_model.wv.vectors[0] += 1
        with pytest.raises(ValueError, match="holds other codes"):
            Average(w2v_model, wv_dtype="int8", wv_mapfile_path=str(mapfile))

    def test_models_saved_before_wv_dtype_load_as_float32(self, w2v_model, sentences, tmp_path):
        model = Average(w2v_model)
        model.train(IndexedList(sentences))
        for name in ("wv_dtype", "wv_codes", "wv_scales", "ngram_codes", "ngram_scales"):
            model.__dict__.pop(name, None)
        path = tmp_path / "old"
        model.save(str(path))
        loaded = Average.load(str(path))
        assert loaded.wv_dtype == BaseSentence2VecModel.wv_dtype == "float32"
        assert loaded.infer(IndexedList(sentences[:2])).shape == (2, 20)

    @pytest.mark.parametrize("fmt", QUANTIZED)
    def test_estimate_memory_counts_the_codes(self, w2v_model, fmt):
        model = Average(w2v_model, wv_dtype=fmt)
        report = model.estimate_memory(max_index=10)
        rows, dim = w2v_model.wv.vectors.shape
        assert report["Word Vector Codes"] == rows * dim * np.dtype(STORAGE_TYPES[fmt]).itemsize
        assert ("Word Vector Scales" in report) == (fmt == "int8")
