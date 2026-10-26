#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Scoring stored float16 and int8 rows: the kernel, SentenceVectors.score and search."""

import numpy as np
import pytest
from pydantic import ValidationError

from fse.inputs import IndexedList
from fse.models import sentencevectors
from fse.models.average_inner import score_rows
from fse.models.sentencevectors import SentenceVectors
from fse.models.sif import SIF

RNG = np.random.default_rng(7)


def _codes(kind, n, dim):
    if kind == "int8":
        return RNG.integers(-127, 128, size=(n, dim), dtype=np.int8)
    return RNG.standard_normal((n, dim), dtype=np.float32).astype(np.float16)


def _reference(codes, scales, weights_t, bias, factors):
    out = codes.astype(np.float64) @ weights_t.astype(np.float64).T
    for per_row in (scales, factors):
        if per_row is not None:
            out *= per_row[:, None]
    if bias is not None:
        out += bias
    return out


class TestKernel:
    @pytest.mark.parametrize("kind", ["int8", "float16"])
    @pytest.mark.parametrize("k", [1, 5])
    @pytest.mark.parametrize("dim", [1, 3, 4, 7, 300])
    @pytest.mark.parametrize("extras", [False, True])
    def test_matches_float64(self, kind, k, dim, extras):
        n = 50
        codes = _codes(kind, n, dim)
        weights_t = RNG.standard_normal((k, dim), dtype=np.float32)
        scales = RNG.random(n, dtype=np.float32) if extras else None
        factors = RNG.random(n, dtype=np.float32) if extras else None
        bias = RNG.standard_normal(k, dtype=np.float32) if extras else None
        out = np.full((n, k), np.nan, dtype=np.float32)
        score_rows(codes, scales, weights_t, bias, factors, out, 0, n)
        expected = _reference(codes, scales, weights_t, bias, factors)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-3)

    def test_only_the_requested_rows_are_written(self):
        codes = _codes("int8", 20, 8)
        weights_t = np.ones((1, 8), dtype=np.float32)
        out = np.full((20, 1), -1, dtype=np.float32)
        score_rows(codes, None, weights_t, None, None, out, 5, 9)
        assert (out[:5] == -1).all()
        assert (out[9:] == -1).all()
        np.testing.assert_array_equal(out[5:9, 0], codes[5:9].sum(axis=1).astype(np.float32))
        score_rows(codes, None, weights_t, None, None, out, 3, 3)

    def test_every_float16_value_is_converted(self):
        values = np.arange(65536, dtype=np.uint16).view(np.float16)
        finite = values[np.isfinite(values)]
        codes = finite.reshape(-1, 1)
        out = np.empty((len(codes), 1), dtype=np.float32)
        score_rows(codes, None, np.ones((1, 1), dtype=np.float32), None, None, out, 0, len(codes))
        np.testing.assert_array_equal(out[:, 0], finite.astype(np.float32))
        special = np.array([[np.inf], [-np.inf], [np.nan]], dtype=np.float16)
        out = np.empty((3, 1), dtype=np.float32)
        score_rows(special, None, np.ones((1, 1), dtype=np.float32), None, None, out, 0, 3)
        assert out[0, 0] == np.inf
        assert out[1, 0] == -np.inf
        assert np.isnan(out[2, 0])

    def test_reads_memmaps_in_place(self, tmp_path):
        codes = np.memmap(tmp_path / "codes", dtype=np.int8, mode="w+", shape=(40, 6))
        codes[:] = _codes("int8", 40, 6)
        weights_t = RNG.standard_normal((2, 6), dtype=np.float32)
        out = np.empty((40, 2), dtype=np.float32)
        score_rows(codes, None, weights_t, None, None, out, 0, 40)
        np.testing.assert_allclose(
            out, _reference(np.asarray(codes), None, weights_t, None, None), rtol=1e-5, atol=1e-4
        )

    @pytest.mark.parametrize(
        ("change", "error", "match"),
        [
            ({"codes": np.zeros((4, 3), dtype=np.float32)}, TypeError, "int8 or float16"),
            ({"codes": np.zeros((4, 6), dtype=np.int8)[:, ::2]}, ValueError, "C-contiguous"),
            ({"codes": np.zeros(4, dtype=np.int8)}, TypeError, "2-dimensional"),
            (
                {"weights_t": np.zeros((1, 4), dtype=np.float32)},
                ValueError,
                "weights_t must have shape",
            ),
            (
                {"weights_t": np.zeros((1, 3), dtype=np.float64)},
                TypeError,
                "weights_t must have dtype",
            ),
            ({"out": np.zeros((4, 2), dtype=np.float32)}, ValueError, "out must have shape"),
            ({"scales": np.zeros(3, dtype=np.float32)}, ValueError, "scales must have shape"),
            ({"bias": np.zeros(2, dtype=np.float32)}, ValueError, "bias must have shape"),
            ({"end": 5}, ValueError, "outside"),
            ({"begin": 3, "end": 2}, ValueError, "outside"),
        ],
    )
    def test_invalid_arguments_raise(self, change, error, match):
        args = {
            "codes": np.zeros((4, 3), dtype=np.int8),
            "scales": None,
            "weights_t": np.zeros((1, 3), dtype=np.float32),
            "bias": None,
            "factors": None,
            "out": np.zeros((4, 1), dtype=np.float32),
            "begin": 0,
            "end": 4,
        }
        args.update(change)
        with pytest.raises(error, match=match):
            score_rows(**args)

    def test_read_only_out_raises(self):
        out = np.zeros((4, 1), dtype=np.float32)
        out.setflags(write=False)
        with pytest.raises(ValueError, match="writeable"):
            score_rows(
                np.zeros((4, 3), np.int8), None, np.zeros((1, 3), np.float32), None, None, out, 0, 4
            )


@pytest.fixture(params=["float32", "float16", "int8"])
def sv(request):
    vectors = SentenceVectors(12)
    vectors.vectors = RNG.standard_normal((70_000, 12), dtype=np.float32)
    vectors.convert(request.param)
    return vectors


class TestScore:
    @pytest.mark.parametrize("matrix", [False, True])
    @pytest.mark.parametrize("bias", [None, "scalar", "vector"])
    def test_matches_the_float32_rows(self, sv, matrix, bias):
        k = 3 if matrix else 1
        weights = RNG.standard_normal((12, k) if matrix else 12)
        b = {None: None, "scalar": 0.5, "vector": RNG.standard_normal(k)}[bias]
        if bias == "vector" and not matrix:
            b = b[:1]
        result = sv.score(weights, b)
        rows = sv.get_vectors().astype(np.float64)
        expected = rows @ weights + (0 if b is None else b)
        assert result.dtype == np.float32
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("workers", [1, 2, 3, 8])
    def test_workers_give_identical_scores(self, sv, workers):
        weights = RNG.standard_normal((12, 2)).astype(np.float32)
        np.testing.assert_array_equal(
            sv.score(weights, workers=workers), sv.score(weights, workers=1)
        )

    def test_out_may_be_a_memmap(self, sv, tmp_path):
        out = np.memmap(tmp_path / "scores", dtype=np.float32, mode="w+", shape=(len(sv), 2))
        weights = RNG.standard_normal((12, 2))
        assert sv.score(weights, out=out) is out
        np.testing.assert_allclose(out, sv.score(weights), rtol=1e-6)

    def test_without_the_extension(self, sv, monkeypatch):
        weights = RNG.standard_normal((12, 2))
        compiled = sv.score(weights, 1.0)
        monkeypatch.setattr(sentencevectors, "score_rows", None)
        np.testing.assert_allclose(sv.score(weights, 1.0), compiled, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"weights": np.ones(12, dtype=np.int64)}, TypeError, "weights must hold floats"),
            ({"weights": np.full(12, np.nan)}, ValueError, "weights must be finite"),
            ({"weights": np.ones(11)}, ValueError, "weights must have shape"),
            ({"weights": np.ones((12, 0))}, ValueError, "weights must have shape"),
            ({"weights": np.ones((12, 2, 1))}, ValueError, "weights must have shape"),
            ({"weights": np.ones((12, 2)), "bias": np.ones(3)}, ValueError, "bias must be"),
            ({"weights": np.ones(12), "bias": np.inf}, ValueError, "bias must be finite"),
            ({"weights": np.ones(12), "out": np.empty(5, np.float32)}, ValueError, "out must be"),
            ({"weights": np.ones(12), "out": np.empty(70_000)}, ValueError, "out must be"),
        ],
    )
    def test_invalid_arguments_raise(self, sv, kwargs, error, match):
        with pytest.raises(error, match=match):
            sv.score(**kwargs)

    def test_workers_are_validated(self, sv):
        with pytest.raises(ValidationError):
            sv.score(np.ones(12), workers=0)

    def test_logistic_regression(self, w2v_model):
        model = SIF(w2v_model, sv_dtype="int8")
        model.train(IndexedList([["the", "cat"], ["a", "dog"], ["the", "dog"]] * 10))
        coef = RNG.standard_normal(model.sv.vector_size)
        probability = 1 / (1 + np.exp(-model.sv.score(coef, -0.1)))
        expected = 1 / (1 + np.exp(-(model.sv.get_vectors() @ coef - 0.1)))
        np.testing.assert_allclose(probability, expected, rtol=1e-5)


@pytest.fixture
def w2v_model(make_w2v):
    return make_w2v(10)


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_search_with_and_without_the_kernel(dtype, monkeypatch):
    sv = SentenceVectors(12)
    sv.vectors = RNG.standard_normal((50_000, 12), dtype=np.float32)
    sv.convert(dtype)
    compiled = sv.most_similar(3, topn=20)
    everything = sv.most_similar(3, topn=None)
    monkeypatch.setattr(sentencevectors, "score_rows", None)
    sv.inv_norms = None
    fallback = sv.most_similar(3, topn=20)
    assert [i for i, _ in fallback] == [i for i, _ in compiled]
    assert [s for _, s in fallback] == pytest.approx([s for _, s in compiled], abs=1e-6)
    np.testing.assert_allclose(sv.most_similar(3, topn=None), everything, atol=1e-6)


def test_int8_scores_do_not_overflow_before_the_scale():
    """127 * 1e37 leaves float32, the scaled result 4e37 does not."""
    sv = SentenceVectors(4)
    sv.vectors = np.ones((3, 4), dtype=np.float32)
    sv.convert("int8")
    result = sv.score(np.full(4, 1e37, dtype=np.float32))
    np.testing.assert_allclose(result, 4e37, rtol=1e-5)
    codes = np.full((1, 4), 127, dtype=np.int8)
    out = np.empty((1, 1), dtype=np.float32)
    score_rows(
        codes,
        np.array([1 / 127], np.float32),
        np.full((1, 4), 1e37, np.float32),
        None,
        None,
        out,
        0,
        1,
    )
    np.testing.assert_allclose(out, 4e37, rtol=1e-5)


def test_weights_beyond_float32_raise():
    sv = SentenceVectors(4)
    sv.vectors = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="weights must be finite in float32"):
        sv.score(np.full(4, 1e100))
    with pytest.raises(ValueError, match="bias must be finite in float32"):
        sv.score(np.ones(4), 1e100)


def test_out_must_not_overlap_inputs():
    sv = SentenceVectors(1)
    sv.vectors = np.ones((5, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="out must not overlap the sentence vectors"):
        sv.score(np.ones(1), out=sv.vectors.reshape(5))
    weights_t = np.ones((1, 2), dtype=np.float32)
    shared = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="out must not overlap weights_t"):
        score_rows(
            np.zeros((2, 2), np.int8),
            None,
            shared[:2].reshape(1, 2),
            None,
            None,
            shared[:2].reshape(2, 1),
            0,
            2,
        )
    with pytest.raises(ValueError, match="out must not overlap scales"):
        score_rows(
            np.zeros((2, 2), np.int8),
            shared[:2],
            weights_t,
            None,
            None,
            shared[:2].reshape(2, 1),
            0,
            2,
        )


def test_search_factors_are_cached():
    import tracemalloc

    sv = SentenceVectors(8)
    sv.vectors = RNG.standard_normal((100_000, 8), dtype=np.float32)
    sv.convert("int8")
    sv.most_similar(0, topn=5, restrict_size=10)
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        sv.most_similar(0, topn=5, restrict_size=10)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        if started:
            tracemalloc.stop()
    # An array of factors for all rows would take 400 KB.
    assert peak < 100_000


def test_a_second_interrupt_waits_for_running_ranges(monkeypatch):
    """After score returns, no range may still write into out."""
    import threading

    release = threading.Event()
    finished = []

    def fake_score_rows(codes, scales, weights_t, bias, factors, out, begin, end):
        if begin == 0:
            raise KeyboardInterrupt
        release.wait(5)
        out[begin:end] = 1
        finished.append(begin)

    original_wait = sentencevectors.wait
    interrupts = []

    def interrupted_wait(futures, *args, **kwargs):
        if not interrupts:
            interrupts.append(1)
            release.set()
            raise KeyboardInterrupt
        return original_wait(futures, *args, **kwargs)

    monkeypatch.setattr(sentencevectors, "score_rows", fake_score_rows)
    monkeypatch.setattr(sentencevectors, "wait", interrupted_wait)
    codes = np.zeros((16384 * 8, 2), dtype=np.int8)
    out = np.zeros((len(codes), 1), dtype=np.float32)
    with sentencevectors._StoredScores(4) as scores, pytest.raises(KeyboardInterrupt):
        scores.run(codes, None, np.ones((1, 2), np.float32), None, None, out)
    snapshot = out.copy()
    started = len(finished)
    release.set()
    np.testing.assert_array_equal(out, snapshot)
    assert len(finished) == started
