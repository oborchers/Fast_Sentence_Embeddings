#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Training float16 and int8 vectors through a ring of float32 windows."""

import logging
import threading
import tracemalloc

import numpy as np
import pytest

from fse import CorpusFile, TextList
from fse.inputs import CIndexedList, IndexedLineDocument, IndexedList
from fse.models import _windowed, sentencevectors, utils
from fse.models._quantize import STORAGE_TYPES, quantize_into
from fse.models._windowed import WindowedWriter
from fse.models.average import Average
from fse.models.sif import SIF
from fse.models.usif import uSIF

# Every test also checks that no file is moved while a memmap still maps it (Windows).
pytestmark = pytest.mark.usefixtures("windows_file_maps")

DIM = 16
MODELS = [(Average, {}), (SIF, {"components": 2}), (uSIF, {"length": 11, "components": 2})]
MODEL_IDS = ["Average", "SIF", "uSIF"]


@pytest.fixture
def w2v(make_w2v):
    return make_w2v(DIM)


@pytest.fixture
def small_windows(monkeypatch):
    """Blocks of 16 rows and windows of 32, so jobs span windows and wait for slots."""
    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 16)
    monkeypatch.setattr(_windowed, "WINDOW_BYTES", 32 * DIM * 4)


@pytest.fixture
def lines(sentences):
    # Several hundred sentences, so the 32-row windows wrap around the ring many times.
    return [" ".join(s) for s in sentences] * 3


def _inputs(kind, lines, tmp_path):
    if kind == "IndexedList":
        return IndexedList([line.split() for line in lines])
    if kind == "TextList":
        return TextList(lines)
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(lines) + "\n")
    return CorpusFile(path) if kind == "CorpusFile" else IndexedLineDocument(path)


def _stored(rows, dtype):
    codes = np.empty(rows.shape, dtype=STORAGE_TYPES[dtype])
    scales = np.empty(len(rows), dtype=np.float32) if dtype == "int8" else None
    quantize_into(rows, codes, scales)
    return codes, scales


def _assert_stored(model, rows):
    codes, scales = _stored(np.asarray(rows), model.sv.dtype)
    assert np.array_equal(np.asarray(model.sv.vectors), codes)
    if scales is not None:
        assert np.array_equal(np.asarray(model.sv.scales), scales)


@pytest.mark.usefixtures("small_windows")
@pytest.mark.parametrize("dtype", ["float16", "int8"])
@pytest.mark.parametrize(("cls", "kwargs"), MODELS, ids=MODEL_IDS)
@pytest.mark.parametrize("kind", ["IndexedList", "TextList", "CorpusFile", "IndexedLineDocument"])
@pytest.mark.parametrize("workers", [1, 2, 8])
def test_windows_store_the_float32_rows_exactly(
    w2v, lines, tmp_path, dtype, cls, kwargs, kind, workers
):
    data = _inputs(kind, lines, tmp_path)
    reference = cls(w2v, workers=workers, **kwargs)
    reference.train(data)
    model = cls(w2v, workers=workers, sv_dtype=dtype, **kwargs)
    # Small jobs, so Python inputs wrap the ring and wait for slots as native ones do.
    model.batch_words = 8
    spy = []
    original = WindowedWriter.__init__

    def record(self, *args, **kw):
        original(self, *args, **kw)
        spy.append(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(WindowedWriter, "__init__", record)
        model.train(data)
    assert spy, "the windowed path did not run"
    if kind in ("IndexedList", "IndexedLineDocument"):
        assert all(len(writer.ring) < writer.n_rows for writer in spy)
        assert all(writer.num_windows > writer.n_windows for writer in spy)
    _assert_stored(model, reference.sv.vectors)
    if cls is not Average:
        np.testing.assert_array_equal(model.svd_res[1], reference.svd_res[1])


@pytest.mark.usefixtures("small_windows")
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize(("cls", "kwargs"), MODELS, ids=MODEL_IDS)
def test_windowed_updates_append_exactly(w2v, lines, tmp_path, mapped, cls, kwargs):
    reference = cls(w2v, **kwargs)
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = cls(w2v, sv_dtype="int8", **kwargs, **path)
    third = len(lines) // 3
    for number, part in enumerate((lines[:third], lines[third : 2 * third], lines[2 * third :])):
        reference.train(TextList(part), update=number > 0)
        model.train(TextList(part), update=number > 0)
    _assert_stored(model, reference.sv.vectors)
    if mapped:
        assert sorted(p.name for p in tmp_path.iterdir()) == ["sv.scales", "sv.vectors"]


@pytest.mark.parametrize(
    ("prepare", "reason"),
    [
        (lambda m, s: CIndexedList(s, custom_index=[i // 2 for i in range(len(s))]), "positions"),
        (lambda m, s: (setattr(m, "_do_train_job", m._do_train_job), IndexedList(s))[1], "kernel"),
        (
            lambda m, s: (setattr(m, "_post_train_calls", m._post_train_calls), IndexedList(s))[1],
            "_post_train_calls is replaced",
        ),
    ],
    ids=["many-to-one", "own-do-train-job", "own-post-train-calls"],
)
def test_other_inputs_use_the_staging_matrix(w2v, sentences, caplog, prepare, reason):
    model = SIF(w2v, sv_dtype="int8")
    data = prepare(model, sentences)
    with caplog.at_level(logging.INFO, logger="fse.models.base_s2v"):
        model.train(data)
    assert any(
        "through a float32 matrix of all new rows" in r.message and reason in r.message
        for r in caplog.records
    )
    reference = SIF(w2v)
    reference.train(data)
    np.testing.assert_allclose(model.sv.get_vectors(), reference.sv.vectors, atol=1e-2)


def test_without_the_extension_the_staging_matrix_is_used(w2v, sentences, monkeypatch, caplog):
    from fse.models import average

    monkeypatch.setattr(average, "CY_ROUTINES", False)
    model = Average(w2v, sv_dtype="float16")
    with caplog.at_level(logging.INFO, logger="fse.models.base_s2v"):
        model.train(IndexedList(sentences))
    assert any("compiled kernel is not used" in r.message for r in caplog.records)


def _training_peak(model, data):
    """Bytes allocated at most while training, beyond what existed before."""
    model.train(TextList(["the"]))  # build the kernel and caches outside the measurement
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        model.train(data)
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        if started:
            tracemalloc.stop()


def test_no_float32_matrix_of_all_rows(w2v, monkeypatch):
    monkeypatch.setattr(_windowed, "WINDOW_BYTES", 1024 * DIM * 4)
    monkeypatch.setattr(utils, "REMOVAL_BLOCK_ROWS", 1024)
    rows = 60_000
    data = TextList(["the product is good"] * rows)
    windowed = _training_peak(Average(w2v, sv_dtype="int8"), data)
    staged = Average(w2v, sv_dtype="int8")
    staged._post_train_calls = staged._post_train_calls  # forces the staging matrix
    staging = _training_peak(staged, data)
    float32_matrix = rows * DIM * 4
    # Staging holds a float32 matrix of all rows besides the codes; windows do not.
    assert staging > float32_matrix
    assert windowed < staging - float32_matrix / 2


@pytest.mark.usefixtures("small_windows")
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failure_while_storing_keeps_the_model(w2v, lines, tmp_path, monkeypatch, mapped, failure):
    path = {"sv_mapfile_path": str(tmp_path / "sv")} if mapped else {}
    model = uSIF(w2v, length=11, components=2, sv_dtype="int8", **path)
    model.train(TextList(lines[:100]))
    codes, scales = np.array(model.sv.vectors), np.array(model.sv.scales)
    state = (model.svd_res[1].copy(), model.svd_weights.copy())
    calls = []
    original = sentencevectors.quantize_into

    def fail_later(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise failure("storing failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(sentencevectors, "quantize_into", fail_later)
    with pytest.raises(failure, match="storing failed"):
        model.train(TextList(lines))  # a new training that learns new components
    assert np.array_equal(model.sv.vectors, codes)
    assert np.array_equal(model.sv.scales, scales)
    assert np.array_equal(model.svd_res[1], state[0])
    assert np.array_equal(model.svd_weights, state[1])
    if mapped:
        assert sorted(p.name for p in tmp_path.iterdir()) == ["sv.scales", "sv.vectors"]
    calls.clear()
    with pytest.raises(failure, match="storing failed"):
        model.train(TextList(lines), update=True)  # an update appending rows
    assert np.array_equal(model.sv.vectors, codes)
    if mapped:
        assert (tmp_path / "sv.vectors").stat().st_size == codes.size


@pytest.mark.usefixtures("small_windows")
def test_a_failing_window_does_not_block_the_producer(w2v, lines, monkeypatch):
    """The producer waits for a free slot; a failed finalizer must release it."""
    model = Average(w2v, sv_dtype="int8", workers=4)

    def fail(*args, **kwargs):
        raise RuntimeError("window failed")

    monkeypatch.setattr(sentencevectors, "quantize_into", fail)
    errors = []

    def run():
        try:
            model.train(TextList(lines * 4))
        except RuntimeError as err:
            errors.append(err)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive(), "training hangs after a failed window"
    assert len(errors) == 1
    assert "window failed" in str(errors[0])


@pytest.mark.usefixtures("small_windows")
def test_float16_overflow_names_the_row(w2v, lines):
    model = Average(w2v, sv_dtype="float16")
    model.train(TextList(lines[:50]))
    big = "fse_overflow_word"
    model.wv.add_vectors([big], np.full((1, DIM), 1e9, dtype=np.float32))
    model.wv.set_vecattr(big, "count", 1)
    model.word_weights = np.ones(len(model.wv), dtype=np.float32)
    data = list(lines[:80])
    data[70] = big
    with pytest.raises(ValueError, match="sentence vector 120 has values beyond the float16"):
        model.train(TextList(data), update=True)
    assert len(model.sv) == 50


def test_ring_sizes():
    writer = WindowedWriter(100, 4, lambda first, rows: None, max_job_rows=10, rows_per_window=32)
    assert writer.num_windows == 4
    assert writer.n_windows == 2
    assert writer.ring.shape == (64, 4)
    # A job longer than a window needs more slots, but never more than there are windows.
    assert (
        WindowedWriter(100, 4, lambda *a: None, max_job_rows=70, rows_per_window=32).n_windows == 4
    )
    assert (
        WindowedWriter(20, 4, lambda *a: None, max_job_rows=10, rows_per_window=32).n_windows == 1
    )


def test_windows_are_finalized_in_order():
    seen = []
    writer = WindowedWriter(
        96,
        2,
        lambda first, rows: seen.append((first, len(rows))),
        max_job_rows=8,
        rows_per_window=32,
    )
    writer.rows_done(40, 96)
    assert seen == []
    writer.rows_done(0, 40)
    assert seen == [(0, 32), (32, 32), (64, 32)]
    writer.finish()
    with pytest.raises(ValueError, match="outside"):
        writer.rows_done(90, 100)


def test_unfinished_windows_raise():
    writer = WindowedWriter(64, 2, lambda *a: None, max_job_rows=8, rows_per_window=32)
    writer.rows_done(0, 32)
    with pytest.raises(RuntimeError, match="only 1 of 2 windows"):
        writer.finish()


def test_a_ring_is_set_up_once(w2v):
    from fse.models.average_inner import KernelContext

    model = Average(w2v)
    model._pre_train_calls()
    context = KernelContext(model, np.zeros((2, DIM), dtype=np.float32))
    context.use_ring(100)
    with pytest.raises(ValueError, match="already a ring"):
        context.use_ring(100)
