"""Linear heads in the runtime: HeadSpec, the fold, make_head and HeadPipeline.infer."""

import dataclasses
import threading

import numpy as np
import pytest

from fse import _reference_head as ref
from fse._backends import UnsupportedByBackendError, get_backend
from fse._contract import Link, Normalize
from fse._runtime import MemorySink, Pipeline
from fse._runtime._head import HeadPipeline, HeadSpec, _buffer, fold, make_head
from fse._stream import StreamSource
from fse._text import make_vocabulary
from support import head as support
from support.encoder import finished
from support.runtime import source, texts

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _spec(dtype="float32", fasttext=False, normalize=False, dim=12, tokenizer="fse_words@1"):
    return finished(dtype, fasttext, normalize, tokenizer=tokenizer, dim=dim)._spec


def _head(spec, k=3, normalized=False, link=Link.SOFTMAX, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((k, spec.output_dim)).astype(F32)
    bias = rng.standard_normal(k).astype(F32)
    return HeadSpec(weight, bias, normalized, link)


def _own(spec, backend="cpu"):
    return Pipeline(spec, backend, vocabulary=make_vocabulary(spec.word_vectors))


def test_head_spec_checks_and_copies():
    weight = np.ones((2, 3), F32)
    head = HeadSpec(weight, np.zeros(2, F32), False, None)
    weight[0, 0] = 5
    assert head.weight[0, 0] == 1
    assert not head.weight.flags.writeable
    assert head.k == 2
    assert head.steps == ()
    assert head.width("logits") == 2
    assert HeadSpec(weight, np.zeros(2, F32), True, Link.OVR).steps == (Normalize(),)
    assert HeadSpec(None, np.zeros(1, F32), False, Link.SIGMOID).width("proba") == 2
    assert head.width("labels") == 1
    with pytest.raises(ValueError, match="no link"):
        head.width("proba")
    for args, match in [
        ((weight, np.zeros(3, F32), False, None), "2 rows, but bias 3"),
        ((weight, np.zeros(2), False, None), "bias must be"),
        ((weight.astype(np.float64), np.zeros(2, F32), False, None), "weight must be"),
        ((np.full((2, 3), np.inf, F32), np.zeros(2, F32), False, None), "weight must be finite"),
        ((None, np.zeros(2, F32), True, None), "cannot be normalized"),
    ]:
        with pytest.raises(ValueError, match=match):
            HeadSpec(*args)
    with pytest.raises(TypeError, match="normalized"):
        HeadSpec(weight, np.zeros(2, F32), 1, None)
    with pytest.raises(TypeError, match="link"):
        HeadSpec(weight, np.zeros(2, F32), False, "softmax@1")


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("workers", [1, 3])
def test_fold_is_the_reference_fold(dtype, workers, monkeypatch):
    spec = _spec(dtype, fasttext=True)
    head = _head(spec)
    monkeypatch.setattr("fse._runtime._head._FOLD_ROWS", 7)  # many blocks
    pool = fold(get_backend("cpu"), spec.pool_input(), head.weight, workers)
    want = support.folded_pool(spec.pool_input(), head.weight)
    np.testing.assert_array_equal(bits(pool.vectors), bits(want[0]))
    np.testing.assert_array_equal(bits(pool.ngram_vectors), bits(want[3]))
    assert pool.scales is None
    assert pool.ngram_scales is None
    assert pool.oov_weight == spec.pool_input().oov_weight
    assert not pool.vectors.flags.writeable


def test_a_fold_that_overflows_names_the_rows(monkeypatch):
    spec = _spec()
    weight = np.full((1, spec.output_dim), 3e38, F32)
    monkeypatch.setattr("fse._runtime._head._FOLD_ROWS", 50)
    with pytest.raises(ValueError, match="does not fold into the table: word rows 0 to 49"):
        fold(get_backend("cpu"), spec.pool_input(), weight, 2)


CASES = [
    # dtype, fasttext, encoder normalizes, head normalized
    ("float32", False, False, False),
    ("float16", True, False, False),
    ("int8", True, True, False),
    ("float32", True, True, True),
    ("int8", False, False, True),
]


@pytest.mark.parametrize(("dtype", "fasttext", "normalize", "normalized"), CASES)
@pytest.mark.parametrize("backend", ["cpu", "numpy"])
def test_logits_are_the_reference(dtype, fasttext, normalize, normalized, backend):
    spec = _spec(dtype, fasttext, normalize)
    head = _head(spec, normalized=normalized)
    sentences = texts(300 if backend == "numpy" else 1500)
    pipeline = make_head(spec, head, backend, own=_own(spec, backend), workers=2)
    assert pipeline.folded == (not normalized)
    got, vectors = pipeline.infer(source("strings", sentences, None), output="logits", workers=3)
    assert vectors is None
    want = support.logits(spec, head.weight, head.bias, sentences, normalized=normalized)
    np.testing.assert_array_equal(bits(got), bits(want))


@pytest.mark.parametrize("kind", ["strings", "stream", "tokens", "file"])
@pytest.mark.parametrize("output", ["logits", "proba", "labels"])
def test_every_source_and_output(kind, output, tmp_path):
    spec = _spec(fasttext=True, tokenizer="external")
    head = _head(spec, k=4, link=Link.OVR)
    sentences = texts(900, seed=4)
    pipeline = make_head(spec, head, "cpu", own=_own(spec))
    got, _ = pipeline.infer(
        source(kind, sentences, tmp_path), output=output, workers=4, chunk_bytes=1 << 16
    )
    z = support.logits(spec, head.weight, head.bias, sentences)
    want = support.outputs(z, output, Link.OVR)
    if output == "labels":
        assert got.dtype == np.int64
        assert got.tolist() == want.tolist()
    else:
        np.testing.assert_array_equal(bits(got), bits(want))


@pytest.mark.parametrize(
    ("normalize", "normalized"), [(True, True), (False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("vector_dtype", ["float32", "int8"])
def test_with_vectors_one_pooling_gives_both_where_the_flags_agree(
    normalize, normalized, vector_dtype
):
    spec = _spec("float16", fasttext=True, normalize=normalize)
    head = _head(spec, k=1, normalized=normalized, link=Link.SIGMOID)
    own = _own(spec)
    pipeline = make_head(spec, head, "cpu", own=own, with_vectors=True)
    assert not pipeline.folded
    assert (pipeline.vectors is None) == (normalize == normalized)
    sentences = texts(700, seed=5)
    got, vectors = pipeline.infer(
        source("strings", sentences, None), output="proba", workers=2, vector_dtype=vector_dtype
    )
    encoded = own.infer(source("strings", sentences, None), workers=1, dtype=vector_dtype)
    if vector_dtype == "int8":
        np.testing.assert_array_equal(vectors.codes, encoded.codes)
        np.testing.assert_array_equal(bits(vectors.scales), bits(encoded.scales))
    else:
        np.testing.assert_array_equal(bits(vectors), bits(encoded))
    z = support.logits(spec, head.weight, head.bias, sentences, normalized=normalized, folded=False)
    np.testing.assert_array_equal(bits(got), bits(ref.proba(z, Link.SIGMOID)))


def test_a_standalone_head_reads_the_folded_table_of_its_spec():
    spec = _spec()
    head = _head(spec, k=2, link=Link.SOFTMAX)
    folded = fold(get_backend("cpu"), spec.pool_input(), head.weight, 1)
    standalone = HeadSpec(None, head.bias, False, Link.SOFTMAX)
    table = Pipeline(spec, "cpu", pool=folded, steps=())  # the table a standalone file holds
    pipeline = HeadPipeline(table, standalone, folded=True)
    sentences = texts(400)
    got, _ = pipeline.infer(source("strings", sentences, None), output="proba", workers=2)
    z = support.logits(spec, head.weight, head.bias, sentences)
    np.testing.assert_array_equal(bits(got), bits(ref.proba(z, Link.SOFTMAX)))
    with pytest.raises(ValueError, match="standalone head has no vectors"):
        make_head(spec, standalone, "cpu", own=_own(spec), with_vectors=True)


def test_outputs_and_vectors_into_sinks_numbered_from_first_row():
    spec = _spec(normalize=True)
    head = _head(spec, k=3, normalized=True)
    pipeline = make_head(spec, head, "cpu", own=_own(spec), with_vectors=True)
    sentences = texts(500)
    out, vec = MemorySink(3), MemorySink(spec.output_dim, "float16")
    out.next_row = vec.next_row = 1000
    got = pipeline.infer(
        source("strings", sentences, None),
        output="proba",
        sink=out,
        vector_sink=vec,
        first_row=1000,
    )
    assert got == (None, None)
    z = support.logits(spec, head.weight, head.bias, sentences, normalized=True)
    np.testing.assert_array_equal(bits(out.result()), bits(ref.proba(z, Link.SOFTMAX)))
    want = support.pooled(spec, sentences, spec.steps).astype(np.float16)
    np.testing.assert_array_equal(vec.result(), want)


def test_outputs_into_a_sink_without_vectors():
    spec = _spec()
    head = _head(spec, k=2)
    pipeline = make_head(spec, head, "cpu", own=_own(spec))
    sentences = texts(300)
    sink = MemorySink(2)
    outputs, vectors = pipeline.infer(
        source("strings", sentences, None), output="logits", sink=sink
    )
    assert outputs is None
    assert vectors is None
    want = support.logits(spec, head.weight, head.bias, sentences)
    np.testing.assert_array_equal(bits(sink.result()), bits(want))


def test_infer_checks_its_arguments_before_reading():
    spec = _spec()
    head = _head(spec, k=2)
    pipeline = make_head(spec, head, "cpu", own=_own(spec))
    read = []

    def stream():
        read.append(1)
        yield "w1"

    for kwargs, error, match in [
        ({"output": "values"}, ValueError, "output must be one of"),
        ({"output": "labels", "sink": MemorySink(1)}, ValueError, "labels go to no sink"),
        ({"output": "logits", "sink": MemorySink(3)}, ValueError, "rows of 2 values"),
        ({"output": "logits", "sink": MemorySink(2, "float16")}, ValueError, "float32 rows"),
        ({"output": "logits", "vector_sink": MemorySink(12)}, ValueError, "made with the vectors"),
        ({"output": "logits", "workers": 0}, ValueError, "workers"),
        ({"output": "logits", "vector_dtype": "int4"}, ValueError, "dtype"),
    ]:
        with pytest.raises(error, match=match):
            pipeline.infer(StreamSource(stream()), **kwargs)
    assert read == []


def test_heads_need_a_backend_that_runs_them(monkeypatch):
    spec = _spec()
    head = _head(spec)
    own = _own(spec)
    caps = own.backend.capabilities()
    monkeypatch.setattr(
        type(own.backend), "capabilities", lambda _self: dataclasses.replace(caps, heads=False)
    )
    with pytest.raises(UnsupportedByBackendError, match="cpu backend cannot run linear heads"):
        make_head(spec, head, "cpu", own=own)
    with pytest.raises(UnsupportedByBackendError, match="cpu backend cannot run linear heads"):
        make_head(spec, HeadSpec(head.weight, head.bias, True, None), "cpu", own=own)


def test_make_head_shares_the_pipeline_of_the_encoder_where_it_can():
    spec = _spec(normalize=True)
    own = _own(spec)
    normalized = make_head(spec, _head(spec, normalized=True), "cpu", own=own)
    assert normalized.reader is own
    assert not normalized.folded
    raw = make_head(spec, _head(spec), "cpu", own=own)
    assert raw.reader is not own
    assert raw.reader.output_dim == 3
    assert raw.reader.vocabulary is own.vocabulary
    with pytest.raises(ValueError, match="own must be the pipeline of spec"):
        make_head(_spec(), _head(spec), "cpu", own=own)


def test_the_window_counts_what_a_head_holds():
    spec = _spec(normalize=False, dim=12)
    folded = make_head(spec, _head(spec, k=3), "cpu", own=_own(spec))
    assert folded._row_bytes("logits", "float32", False) == 4 * 3 + 4 * 3 + 4 * 3
    assert folded._row_bytes("labels", "float32", False) == 4 * 3 + 4 * 3 + 8
    normalized = make_head(
        spec, _head(spec, k=3, normalized=True), "cpu", own=_own(spec), with_vectors=True
    )
    vectors = normalized.vectors._row_bytes("int8", True)
    assert (
        normalized._row_bytes("proba", "int8", True) == 4 * 12 + 12 * 12 + 4 * 3 + 4 * 3 + vectors
    )


def test_worker_buffers_are_reused_per_shape():
    local = threading.local()
    a = _buffer(local, "x", 10, 4)
    assert _buffer(local, "x", 7, 4).base is a.base
    assert _buffer(local, "x", 7, 5).shape == (7, 5)
    assert _buffer(local, "x", 20, 5).shape == (20, 5)


def test_errors_name_the_rows_of_the_run():
    spec = _spec(normalize=True)
    row = support.pooled(spec, ["w1"], spec.steps)[0]
    weight = (np.sign(row) * F32(3e38)).astype(F32)[None, :]  # its dot product overflows
    head = HeadSpec(weight, np.zeros(1, F32), True, None)
    pipeline = make_head(spec, head, "cpu", own=_own(spec))
    sink = MemorySink(1)
    sink.next_row = 50
    with pytest.raises(ValueError, match="row 51 are not finite"):
        pipeline.infer(
            source("strings", ["", "w1"], None), output="logits", sink=sink, first_row=50
        )
