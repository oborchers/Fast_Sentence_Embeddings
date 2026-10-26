#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""float16 and int8 sentence vectors: the arithmetic and the feature end to end."""

import os
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from fse.inputs import CIndexedList, IndexedList
from fse.models import _quantize, _windowed, sentencevectors
from fse.models._quantize import dequantize, quantize_into
from fse.models.average import Average
from fse.models.base_s2v import EPS
from fse.models.sentencevectors import SentenceVectors
from fse.models.sif import SIF
from fse.models.usif import uSIF

# Every test also checks that no file is moved while a memmap still maps it (Windows).
pytestmark = pytest.mark.usefixtures("windows_file_maps")

DIM = 20
QUANTIZED = ["float16", "int8"]


@pytest.fixture
def w2v(make_w2v):
    return make_w2v(DIM)


def _stored(rows, dtype):
    """Codes and scales as quantize_into stores them."""
    codes = np.empty(rows.shape, dtype=_quantize.STORAGE_TYPES[dtype])
    scales = np.empty(len(rows), dtype=np.float32) if dtype == "int8" else None
    quantize_into(rows, codes, scales)
    return codes, scales


# --- The arithmetic -----------------------------------------------------------------------


class TestArithmetic:
    def test_int8_round_trip_is_within_half_a_step(self):
        rows = np.random.default_rng(0).standard_normal((100, 30)).astype(np.float32)
        codes, scales = _stored(rows, "int8")
        assert codes.dtype == np.int8
        assert np.abs(codes).max() == 127
        np.testing.assert_allclose(scales, np.abs(rows).max(axis=1) / 127, rtol=1e-6)
        error = np.abs(dequantize(codes, scales) - rows)
        assert (error <= scales[:, None] / 2 + 1e-7).all()

    def test_cosines_of_int8_codes_equal_those_of_the_rows(self):
        rows = np.random.default_rng(1).standard_normal((50, 16)).astype(np.float32)
        codes, scales = _stored(rows, "int8")
        restored = dequantize(codes, scales)

        def cos(x):
            x = x.astype(np.float64)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            return x @ x.T

        np.testing.assert_allclose(cos(codes), cos(restored), atol=1e-6)

    def test_zero_and_eps_rows_round_trip(self):
        rows = np.zeros((3, 4), dtype=np.float32)
        rows[1] = EPS
        rows[2, 0] = -EPS
        for dtype in QUANTIZED:
            codes, scales = _stored(rows, dtype)
            restored = dequantize(codes, scales)
            assert not restored[0].any()
            np.testing.assert_allclose(restored[1:], rows[1:], rtol=1e-3)

    def test_float16_out_of_range_raises(self):
        rows = np.ones((5, 3), dtype=np.float32)
        rows[3, 1] = 1e6
        with pytest.raises(ValueError, match=r"sentence vector 13 has values beyond the float16"):
            quantize_into(rows, np.empty((5, 3), dtype=np.float16), None, first_row=10)

    @pytest.mark.parametrize("dtype", QUANTIZED)
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_rows_raise(self, dtype, value):
        rows = np.ones((4, 3), dtype=np.float32)
        rows[2, 0] = value
        codes = np.empty(rows.shape, dtype=_quantize.STORAGE_TYPES[dtype])
        scales = np.empty(4, dtype=np.float32) if dtype == "int8" else None
        with pytest.raises(ValueError, match="sentence vector 2 contains NaN or infinite"):
            quantize_into(rows, codes, scales)

    def test_blocks_give_the_same_codes(self, monkeypatch):
        rows = np.random.default_rng(2).standard_normal((50, 8)).astype(np.float32)
        whole = _stored(rows, "int8")
        monkeypatch.setattr(_quantize, "REMOVAL_BLOCK_ROWS", 7)
        blocked = _stored(rows, "int8")
        assert np.array_equal(whole[0], blocked[0])
        assert np.array_equal(whole[1], blocked[1])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="cannot hold rows"):
            quantize_into(np.ones((2, 3), np.float32), np.empty((2, 4), np.float16), None)


# --- Training end to end ------------------------------------------------------------------


def _expected(cls, w2v, data, **kwargs):
    """The float32 training of the same data, stored in the format afterwards."""
    model = cls(w2v, **kwargs)
    model.train(data)
    return model


MODELS = [(Average, {}), (SIF, {"components": 2}), (uSIF, {"length": 11, "components": 2})]
MODEL_IDS = ["Average", "SIF", "uSIF"]


@pytest.mark.parametrize("dtype", ["float32", *QUANTIZED])
@pytest.mark.parametrize("mapped", [False, True])
def test_reads_are_copies(w2v, sentences, tmp_path, dtype, mapped):
    """A read never aliases the stored vectors, also for float32 memmaps."""
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = SIF(w2v, sv_dtype=dtype, **path)
    model.train(IndexedList(sentences))
    stored = np.array(model.sv.vectors)
    for rows in (model.sv.get_vectors(), next(model.sv.iter_blocks())[1]):
        assert not np.shares_memory(rows, model.sv.vectors)
        rows += 1
    assert np.array_equal(np.asarray(model.sv.vectors), stored)


@pytest.mark.parametrize("dtype", QUANTIZED)
@pytest.mark.parametrize(("cls", "kwargs"), MODELS, ids=MODEL_IDS)
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("indices", ["identity", "many-to-one"])
def test_training_stores_the_float32_vectors_exactly(
    w2v, sentences, tmp_path, dtype, cls, kwargs, mapped, workers, indices
):
    data = (
        IndexedList(sentences)
        if indices == "identity"
        else CIndexedList(sentences, custom_index=[i // 2 for i in range(len(sentences))])
    )
    reference = _expected(cls, w2v, data, workers=1, **kwargs)
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = cls(w2v, sv_dtype=dtype, workers=workers, **kwargs, **path)
    model.train(data)
    codes, scales = _stored(np.asarray(reference.sv.vectors), dtype)
    if indices == "identity" or workers == 1:
        assert np.array_equal(np.asarray(model.sv.vectors), codes)
        if scales is not None:
            assert np.array_equal(np.asarray(model.sv.scales), scales)
    else:
        # Rows shared by several sentences are added in varying order across workers.
        np.testing.assert_allclose(
            model.sv.get_vectors(), dequantize(codes, scales), rtol=1e-2, atol=1e-6
        )
    assert model.sv.vectors.dtype == _quantize.STORAGE_TYPES[dtype]
    if mapped:
        assert isinstance(model.sv.vectors, np.memmap)
        assert not (tmp_path / "sv.staging.vectors").exists()
        assert (tmp_path / "sv.scales").exists() is (dtype == "int8")


def test_float32_stays_the_default(w2v, sentences):
    model = SIF(w2v)
    assert model.sv.dtype == "float32"
    assert not model.sv.quantized
    model.train(IndexedList(sentences))
    assert model.sv.vectors.dtype == np.float32
    assert model.sv.scales is None


@pytest.mark.parametrize("dtype", QUANTIZED)
@pytest.mark.parametrize(("cls", "kwargs"), MODELS, ids=MODEL_IDS)
@pytest.mark.parametrize("mapped", [False, True])
def test_updates_append_exactly(w2v, sentences, tmp_path, dtype, cls, kwargs, mapped):
    first, second = IndexedList(sentences[:40]), IndexedList(sentences[40:70])
    reference = cls(w2v, **kwargs)
    reference.train(first)
    reference.train(second, update=True)
    reference.train(second, update=True)
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = cls(w2v, sv_dtype=dtype, **kwargs, **path)
    model.train(first)
    stored_before = np.array(model.sv.vectors[:40])
    model.train(second, update=True)
    model.train(second, update=True)
    codes, scales = _stored(np.asarray(reference.sv.vectors), dtype)
    assert len(model.sv) == 100
    assert np.array_equal(np.asarray(model.sv.vectors), codes)
    if scales is not None:
        assert np.array_equal(np.asarray(model.sv.scales), scales)
    assert np.array_equal(np.asarray(model.sv.vectors[:40]), stored_before)


def test_update_needing_new_components_raises(w2v, sentences):
    model = SIF(w2v, components=0, sv_dtype="int8")
    model.train(IndexedList(sentences[:40]))
    codes = np.array(model.sv.vectors)
    model.components = 2
    with pytest.raises(ValueError, match="compute new principal components from the existing"):
        model.train(IndexedList(sentences[40:]), update=True)
    assert np.array_equal(model.sv.vectors, codes)


@pytest.mark.parametrize("mapped", [False, True])
def test_failed_training_keeps_the_stored_rows(w2v, sentences, tmp_path, mapped):
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = SIF(w2v, sv_dtype="int8", **path)
    model.train(IndexedList(sentences[:40]))
    codes, scales = np.array(model.sv.vectors), np.array(model.sv.scales)

    def fail(**kwargs):
        raise RuntimeError("post-train failed")

    model._post_train_calls = fail
    with pytest.raises(RuntimeError, match="post-train failed"):
        model.train(IndexedList(sentences[40:]), update=True)
    assert np.array_equal(model.sv.vectors, codes)
    assert np.array_equal(model.sv.scales, scales)
    assert not (tmp_path / "sv.staging.vectors").exists()


def test_failed_staging_keeps_the_components(w2v, sentences, monkeypatch):
    """Staging: components learned before storing fails are dropped with the rows."""
    model = SIF(w2v, components=2, sv_dtype="int8")
    model.train(CIndexedList(sentences, custom_index=[i // 2 for i in range(len(sentences))]))
    before = model.svd_res[1].copy()

    def fail(*args, **kwargs):
        raise RuntimeError("storing failed")

    monkeypatch.setattr(sentencevectors, "quantize_into", fail)
    with pytest.raises(RuntimeError, match="storing failed"):
        model.train(CIndexedList(sentences[::-1], custom_index=list(range(len(sentences)))[::-1]))
    np.testing.assert_array_equal(model.svd_res[1], before)


@pytest.mark.parametrize("mapped", [False, True])
def test_float16_overflow_raises_and_keeps_the_rows(w2v, sentences, tmp_path, mapped):
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = Average(w2v, sv_dtype="float16", **path)
    model.train(IndexedList(sentences[:40]))
    codes = np.array(model.sv.vectors)
    model.wv.vectors = model.wv.vectors * np.float32(1e9)
    with pytest.raises(ValueError, match="beyond the float16 range"):
        model.train(IndexedList(sentences[40:]), update=True)
    assert np.array_equal(model.sv.vectors, codes)
    if mapped:
        expected = codes.size * 2
        assert (tmp_path / "sv.vectors").stat().st_size == expected
        assert not (tmp_path / "sv.staging.vectors").exists()


def test_invalid_dtype_raises(w2v):
    with pytest.raises(ValidationError, match="sv_dtype"):
        SIF(w2v, sv_dtype="int4")
    with pytest.raises(ValidationError):
        SentenceVectors(10, dtype="float64")


def test_infer_returns_float32(w2v, sentences):
    model = uSIF(w2v, length=11, sv_dtype="int8")
    model.train(IndexedList(sentences))
    reference = uSIF(w2v, length=11)
    reference.train(IndexedList(sentences))
    output = model.infer(IndexedList(sentences[:5]))
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, reference.infer(IndexedList(sentences[:5])))


def test_memory_estimate_reports_the_format(w2v):
    model = SIF(w2v, sv_dtype="int8")
    report = model.estimate_memory(max_index=1000)
    assert report["Sentence Vectors"] == 1000 * DIM
    assert report["Sentence Vector Scales"] == 4000
    assert report["Staging"] == 1000 * DIM * 4


# --- Saving, loading, converting ----------------------------------------------------------


@pytest.mark.parametrize("dtype", QUANTIZED)
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize(("cls", "kwargs"), MODELS, ids=MODEL_IDS)
def test_save_and_load(w2v, sentences, tmp_path, dtype, mapped, cls, kwargs):
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = cls(w2v, sv_dtype=dtype, **kwargs, **path)
    model.train(IndexedList(sentences))
    expected = model.sv.most_similar(0, topn=5)
    model.save(str(tmp_path / "model"))
    loaded = cls.load(str(tmp_path / "model"))
    assert loaded.sv.dtype == dtype
    assert np.array_equal(np.asarray(loaded.sv.vectors), np.asarray(model.sv.vectors))
    if dtype == "int8":
        assert np.array_equal(np.asarray(loaded.sv.scales), np.asarray(model.sv.scales))
    assert loaded.sv.most_similar(0, topn=5) == expected
    loaded.train(IndexedList(sentences[:10]), update=True)
    assert len(loaded.sv) == len(sentences) + 10


@pytest.mark.parametrize("damage", ["truncated", "missing"])
def test_damaged_scales_raise_on_load(w2v, sentences, tmp_path, damage):
    model = SIF(w2v, sv_dtype="int8", sv_mapfile_path=str(tmp_path / "sv"))
    model.train(IndexedList(sentences))
    model.save(str(tmp_path / "model"))
    del model
    scales = tmp_path / "sv.scales"
    if damage == "missing":
        scales.unlink()
        error, match = FileNotFoundError, "is missing"
    else:
        scales.write_bytes(scales.read_bytes()[:-4])
        error, match = ValueError, r"has .* bytes, but sentence vectors"
    with pytest.raises(error, match=match):
        SIF.load(str(tmp_path / "model"))


def test_pickles_without_a_format_load_as_float32(tmp_path):
    sv = SentenceVectors(4)
    sv.vectors = np.ones((3, 4), dtype=np.float32)
    state = dict(sv.__dict__)
    for attr in ("dtype", "scales", "inv_norms"):
        del state[attr]
    old = SentenceVectors.__new__(SentenceVectors)
    old.__setstate__(state)
    assert old.dtype == "float32"
    assert old.scales is None
    assert old.most_similar(0, topn=1) == [(1, pytest.approx(1.0))]


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize(
    ("source", "target"),
    [("float32", "int8"), ("float32", "float16"), ("int8", "float16"), ("float16", "float32")],
)
def test_convert(w2v, sentences, tmp_path, mapped, source, target):
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = SIF(w2v, sv_dtype=source, **path)
    model.train(IndexedList(sentences))
    before = model.sv.get_vectors()
    model.sv.convert(target)
    assert model.sv.dtype == target
    codes, scales = _stored(before, target)
    assert np.array_equal(np.asarray(model.sv.vectors), codes)
    if scales is not None:
        assert np.array_equal(np.asarray(model.sv.scales), scales)
    if mapped:
        itemsize = np.dtype(_quantize.STORAGE_TYPES[target]).itemsize
        assert (tmp_path / "sv.vectors").stat().st_size == before.size * itemsize
        assert (tmp_path / "sv.scales").exists() is (target == "int8")
        assert not list(tmp_path.glob("*.converting"))
    # Converting keeps the model usable, also across save and load.
    model.save(str(tmp_path / "model"))
    loaded = SIF.load(str(tmp_path / "model"))
    np.testing.assert_array_equal(loaded.sv.get_vectors(), model.sv.get_vectors())
    model.sv.convert(target)  # the same format again changes nothing


def test_convert_rejects_unknown_formats():
    with pytest.raises(ValidationError):
        SentenceVectors(4).convert("int4")


# --- Reading and searching ----------------------------------------------------------------


@pytest.fixture(params=QUANTIZED)
def trained(request, w2v, sentences):
    model = SIF(w2v, sv_dtype=request.param)
    model.train(IndexedList(sentences))
    return model


def _float32_copy(sv):
    copy = SentenceVectors(sv.vector_size)
    copy.vectors = sv.get_vectors()
    return copy


class TestReading:
    def test_reads_return_float32(self, trained):
        rows = trained.sv.get_vectors()
        assert rows.dtype == np.float32
        np.testing.assert_array_equal(trained.sv[3], rows[3])
        np.testing.assert_array_equal(trained.sv[[1, 4]], rows[[1, 4]])
        assert trained.sv.get_vector(3).dtype == np.float32
        assert not trained.sv.get_vector(3).flags.writeable

    @pytest.mark.parametrize("block_rows", [1, 7, 10_000])
    def test_iter_blocks_covers_every_row(self, trained, block_rows):
        blocks = list(trained.sv.iter_blocks(block_rows))
        assert [begin for begin, _ in blocks] == list(range(0, len(trained.sv), block_rows))
        np.testing.assert_array_equal(
            np.concatenate([block for _, block in blocks]), trained.sv.get_vectors()
        )

    @pytest.mark.parametrize("block_rows", [0, -1, True, 2.0])
    def test_iter_blocks_rejects_invalid_sizes(self, trained, block_rows):
        with pytest.raises((ValueError, ValidationError)):
            list(trained.sv.iter_blocks(block_rows))

    def test_similarity_uses_the_float32_rows(self, trained):
        copy = _float32_copy(trained.sv)
        assert trained.sv.similarity(2, 5) == pytest.approx(copy.similarity(2, 5), abs=1e-6)

    def test_normalized_vectors(self, trained):
        np.testing.assert_allclose(
            np.linalg.norm(trained.sv.get_vector(3, use_norm=True)), 1, rtol=1e-5
        )
        assert trained.sv.vectors_norm is None

    @pytest.mark.parametrize("restrict_size", [None, 50, (20, 80)])
    def test_most_similar_matches_the_float32_search(self, trained, restrict_size):
        copy = _float32_copy(trained.sv)
        for query in (0, [3, 7]):
            got = trained.sv.most_similar(positive=query, topn=10, restrict_size=restrict_size)
            want = copy.most_similar(positive=query, topn=10, restrict_size=restrict_size)
            assert [i for i, _ in got] == [i for i, _ in want]
            assert [s for _, s in got] == pytest.approx([s for _, s in want], abs=1e-5)
        everything = trained.sv.most_similar(positive=0, topn=None)
        np.testing.assert_allclose(everything, copy.most_similar(positive=0, topn=None), atol=1e-5)

    def test_similar_by_sentence(self, trained):
        result = trained.sv.similar_by_sentence(sentence=trained.wv.index_to_key[:3], model=trained)
        assert len(result) == 10

    def test_init_sims_replace_normalizes_the_stored_rows(self, trained):
        trained.sv.init_sims(replace=True)
        norms = np.linalg.norm(trained.sv.get_vectors(), axis=1)
        np.testing.assert_allclose(norms[norms > 0], 1, rtol=2e-3)
        assert trained.sv.vectors.dtype == _quantize.STORAGE_TYPES[trained.sv.dtype]


def test_search_of_mapped_vectors_writes_no_normalized_copy(w2v, sentences, tmp_path):
    model = SIF(w2v, sv_dtype="int8", sv_mapfile_path=str(tmp_path / "sv"))
    model.train(IndexedList(sentences))
    model.sv.most_similar(0, topn=5)
    assert not (tmp_path / "sv.vectors_norm").exists()
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        model.sv.most_similar(0, topn=5)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        if started:
            tracemalloc.stop()
    # One block of float32 rows at most, not a copy per sentence beyond it.
    assert peak < 2 * min(len(model.sv), 16384) * DIM * 4 + 65536


def test_files_of_float32_vectors_keep_their_names(w2v, sentences, tmp_path):
    model = SIF(w2v, sv_mapfile_path=str(tmp_path / "sv"))
    model.train(IndexedList(sentences))
    assert sorted(os.listdir(tmp_path)) == ["sv.vectors"]
    assert Path(model.sv.vectors.filename) == (tmp_path / "sv.vectors").absolute()


def test_old_usif_model_converts_and_keeps_its_weighting(w2v, sentences, tmp_path):
    """A uSIF model as fse 1.0 saved it keeps lambda squared through convert and updates.

    fse 1.0 removed each component weighted by lambda squared and stored the unsquared
    weights; loading squares them, so infer and appended rows match the stored ones.
    """
    from fse.models.utils import remove_principal_components

    first, second = IndexedList(sentences[:60]), IndexedList(sentences[60:])
    model = uSIF(w2v, length=11, components=2)
    model.train(first)
    raw = uSIF(w2v, length=11, components=0)
    raw.train(first)
    weights = model.svd_weights.copy()
    legacy_rows = raw.sv.vectors.copy()
    remove_principal_components(legacy_rows, svd_res=model.svd_res, weights=weights**2)
    model.sv.vectors = legacy_rows
    del model.component_weighting  # as fse 1.0 saved it
    model.save(str(tmp_path / "old"))

    old = uSIF.load(str(tmp_path / "old"))
    assert old.component_weighting == "lambda_squared"
    # Training rows start at EPS, inferred ones at zero.
    np.testing.assert_allclose(old.infer(first), legacy_rows, rtol=1e-5, atol=3e-7)
    old.sv.convert("int8")
    old.train(second, update=True)
    old.save(str(tmp_path / "converted"))
    loaded = uSIF.load(str(tmp_path / "converted"))
    assert loaded.component_weighting == "lambda_squared"
    np.testing.assert_array_equal(loaded.svd_weights, weights**2)
    # The same update of the old model in float32, stored afterwards.
    reference = uSIF.load(str(tmp_path / "old"))
    reference.train(second, update=True)
    codes, scales = _stored(np.asarray(reference.sv.vectors[60:]), "int8")
    assert np.array_equal(np.asarray(loaded.sv.vectors[60:]), codes)
    assert np.array_equal(np.asarray(loaded.sv.scales[60:]), scales)
    np.testing.assert_allclose(
        loaded.sv.get_vectors(0, 60), legacy_rows, rtol=0, atol=np.abs(legacy_rows).max() / 127
    )


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_get_vectors_takes_integers(trained, bad):
    with pytest.raises(TypeError, match="must be an int"):
        trained.sv.get_vectors(bad)
    np.testing.assert_array_equal(trained.sv.get_vectors(-2), trained.sv.get_vectors()[-2:])
    np.testing.assert_array_equal(
        trained.sv.get_vectors(np.int64(1), 3), trained.sv.get_vectors()[1:3]
    )


def test_failed_file_replacement_restores_the_vectors(w2v, sentences, tmp_path, monkeypatch):
    model = SIF(w2v, sv_mapfile_path=str(tmp_path / "sv"))
    model.train(IndexedList(sentences))
    before = model.sv.get_vectors()
    original = os.replace
    calls = []

    def fail_second_written_file(src, dst):
        calls.append((src, dst))
        if str(src).endswith(".scales.converting"):
            raise OSError("disk full")
        return original(src, dst)

    monkeypatch.setattr(os, "replace", fail_second_written_file)
    with pytest.raises(OSError, match="disk full"):
        model.sv.convert("int8")
    monkeypatch.setattr(os, "replace", original)
    assert model.sv.dtype == "float32"
    np.testing.assert_array_equal(model.sv.get_vectors(), before)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sv.vectors"]
    model.sv.convert("int8")  # works once the disk is fine again
    assert model.sv.dtype == "int8"


def test_failed_writer_creation_cuts_grown_files_back(w2v, sentences, tmp_path, monkeypatch):
    model = SIF(w2v, sv_dtype="int8", sv_mapfile_path=str(tmp_path / "sv"))
    model.train(IndexedList(sentences[:50]))
    sizes = {p.name: p.stat().st_size for p in tmp_path.iterdir()}
    codes, scales = np.array(model.sv.vectors), np.array(model.sv.scales)
    expected = model.sv.most_similar(0, topn=5)
    original = sentencevectors.np_memmap
    failures = []

    def fail_once_on_scales(path, *args, **kwargs):
        # Only the first map of the grown scales fails, as when the disk is briefly full.
        if str(path).endswith(".scales") and kwargs.get("mode") == "r+" and not failures:
            failures.append(path)
            raise OSError("cannot map")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(sentencevectors, "np_memmap", fail_once_on_scales)
    with pytest.raises(OSError, match="cannot map"):
        model.train(IndexedList(sentences[50:]), update=True)
    monkeypatch.setattr(sentencevectors, "np_memmap", original)
    assert {p.name: p.stat().st_size for p in tmp_path.iterdir()} == sizes
    assert np.array_equal(model.sv.vectors, codes)
    assert np.array_equal(model.sv.scales, scales)
    assert model.sv.most_similar(0, topn=5) == expected
    model.train(IndexedList(sentences[50:]), update=True)  # usable afterwards
    assert len(model.sv) == len(sentences)


def test_replaced_pre_train_calls_use_staging(w2v, sentences, caplog):
    import logging

    model = SIF(w2v, sv_dtype="int8")
    model._pre_train_calls = model._pre_train_calls
    with caplog.at_level(logging.INFO, logger="fse.models.base_s2v"):
        model.train(IndexedList(sentences))
    assert any("_pre_train_calls is replaced" in r.message for r in caplog.records)


def test_int8_scales_reconstruct_finite_values():
    rows = np.array([[np.finfo(np.float32).max, 1.0], [-np.finfo(np.float32).max, 0.0]], np.float32)
    codes, scales = _stored(rows, "int8")
    restored = dequantize(codes, scales)
    assert np.isfinite(restored).all()
    np.testing.assert_allclose(restored[:, 0], rows[:, 0], rtol=1e-6)


def test_tiny_int8_vectors_have_a_direction():
    sv = SentenceVectors(3)
    sv.vectors = np.array([[1e-38, 2e-38, 0], [1, 2, 0]], dtype=np.float32)
    sv.convert("int8")
    unit = sv.get_vector(0, use_norm=True)
    assert np.isfinite(unit).all()
    np.testing.assert_allclose(np.linalg.norm(unit), 1, rtol=1e-6)
    # 1e-38 / (2e-38 / 127) = 63.5 rounds to 64, which turns the direction a little.
    assert sv.most_similar(1, topn=1)[0][1] == pytest.approx(1.0, abs=1e-4)


def test_first_training_into_a_mapfile_rolls_back(w2v, sentences, tmp_path, monkeypatch):
    model = SIF(w2v, sv_dtype="int8", sv_mapfile_path=str(tmp_path / "sv"))
    original = os.replace

    def fail_on_scales(src, dst):
        if str(dst).endswith(".scales"):
            raise OSError("disk full")
        return original(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_scales)
    with pytest.raises(OSError, match="disk full"):
        model.train(IndexedList(sentences))
    monkeypatch.setattr(os, "replace", original)
    assert list(tmp_path.iterdir()) == []
    assert len(model.sv) == 0
    model.train(IndexedList(sentences))
    assert len(model.sv) == len(sentences)


def test_failed_staging_allocation_keeps_the_model(w2v, sentences, tmp_path, monkeypatch):
    model = SIF(w2v, sv_dtype="int8", sv_mapfile_path=str(tmp_path / "sv"))
    model.train(IndexedList(sentences[:50]))
    codes = np.array(model.sv.vectors)
    original = sentencevectors.np_memmap

    def no_space(path, *args, **kwargs):
        if str(path).endswith(".staging.vectors"):
            raise OSError("no space left")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(sentencevectors, "np_memmap", no_space)
    data = CIndexedList(sentences[50:], custom_index=list(range(len(sentences) - 50))[::-1])
    with pytest.raises(OSError, match="no space left"):
        model.train(data, update=True)
    assert "_saved" not in model.sv.__dict__
    assert np.array_equal(model.sv.vectors, codes)
    assert not (tmp_path / "sv.staging.vectors").exists()


def test_score_rejects_a_second_map_of_the_mapfile(w2v, sentences, tmp_path):
    model = SIF(w2v, sv_mapfile_path=str(tmp_path / "sv"))
    model.train(IndexedList(sentences))
    other = np.memmap(tmp_path / "sv.vectors", dtype=np.float32, mode="r+", shape=(len(sentences),))
    with pytest.raises(ValueError, match="out must not be a map of the mapfile"):
        model.sv.score(np.ones(DIM), out=other)


def test_a_hook_borrowed_from_another_model_uses_staging(w2v, sentences, caplog):
    import logging

    class Borrowed(SIF):
        _post_train_calls = Average._post_train_calls

    model = Borrowed(w2v, sv_dtype="int8")
    with caplog.at_level(logging.INFO, logger="fse.models.base_s2v"):
        model.train(IndexedList(sentences))
    assert any("_post_train_calls is replaced" in r.message for r in caplog.records)


def test_memory_estimate_of_windowed_training(w2v, sentences):
    model = SIF(w2v, sv_dtype="int8")
    rows = 200_000_000
    windowed = model.estimate_memory(max_index=rows, staging_reason=None)
    staged = model.estimate_memory(max_index=rows, staging_reason="custom indices")
    assert staged["Staging"] == rows * DIM * 4
    # Two windows of about WINDOW_BYTES each, whatever the number of rows.
    assert windowed["Staging"] <= 2 * _windowed.WINDOW_BYTES
    assert windowed["Staging"] < staged["Staging"] / 10
