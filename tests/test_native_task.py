"""Tests of the part _native_task.pxi of fse._native: pool_text, stages 1 to 4 of a range
of text in one section without the GIL, against the calls it replaces (TextLookup.encode,
the Batch, CpuBackend.rows) one after the other: the same rows, bits and errors."""

import numpy as np
import pytest

from fse import _native
from fse._contract import OOV_TOKEN, Normalize
from fse._runtime import EncodeSpec, Pipeline
from fse._text import ArrowSource, StringSource, new_workspace
from fse._wordvectors import WordVectors
from support.arrow_c import Producer
from support.runtime import texts, word_vectors

F32 = np.float32


def _pipeline(dtype="float32", fasttext=False, normalize=True, wv=None):
    wv = wv or word_vectors(dtype, fasttext=fasttext)
    steps = (Normalize(),) if normalize else ()
    ft = {"oov_weight": 0.5, "max_ngrams": 4} if wv.fasttext is not None else {}
    return Pipeline(EncodeSpec(wv, np.ones(len(wv), F32), steps, **ft))


def _text(sentences):
    source = StringSource(sentences)
    return (source._array(), source._offsets)


def _fused(pipeline, text, first, last, rows, first_row=0, out=None, after=None):
    out = np.zeros((rows, pipeline.output_dim), F32) if out is None else out
    _native.pool_text(
        pipeline.prepared.state.kernel,
        pipeline.lookup,
        new_workspace(),
        text,
        first,
        last,
        rows,
        first_row,
        out,
        bool(pipeline.prepared.steps),
        after,
    )
    return out


def _sequence(pipeline, text, first, last, rows, first_row=0, out=None, after=None):
    """The calls pool_text replaces, one after the other, as the runtime made them."""
    out = np.zeros((rows, pipeline.output_dim), F32) if out is None else out
    ws = new_workspace()
    if isinstance(text, _native.ArrowColumn):
        batch = text.encode(pipeline.lookup, first, last, first_row, ws)
    else:
        batch = pipeline.lookup.encode(text[0], text[1], first, last, first_row, ws)
    if batch.n_sentences != rows:
        raise RuntimeError("the source changed since it was scanned")
    if after is not None:
        after()
    pipeline.backend.rows(batch, pipeline.prepared, out)
    return out


def _outcome(call, *args, **kwargs):
    try:
        return call(*args, **kwargs).view(np.uint32).tobytes()
    except Exception as error:  # noqa: BLE001 - compared by type and message
        return type(error), str(error)


def _same(pipeline, text, first, last, rows, **kwargs):
    got = _outcome(_fused, pipeline, text, first, last, rows, **kwargs)
    want = _outcome(_sequence, pipeline, text, first, last, rows, **kwargs)
    assert got == want
    return got


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_the_rows_are_those_of_the_calls_it_replaces(dtype, fasttext, normalize):
    pipeline = _pipeline(dtype, fasttext, normalize)
    sentences = texts(400)
    text = _text(sentences)
    got = _same(pipeline, text, 0, 400, 400)
    assert isinstance(got, bytes)
    _same(pipeline, text, 17, 230, 213, first_row=1000)
    _same(pipeline, text, 5, 5, 0)


def test_lines_and_arrow_columns_give_the_same_rows():
    pipeline = _pipeline()
    sentences = texts(300)
    data = np.frombuffer("".join(s + "\n" for s in sentences).encode(), np.uint8)
    lines = _same(pipeline, (data, None), 0, len(data), 300)
    p = Producer()
    column = ArrowSource(_Strings(p, sentences))._column
    assert _same(pipeline, column, 0, 300, 300) == lines
    assert (
        _same(pipeline, column, 40, 90, 50)
        == _fused(pipeline, _text(sentences), 40, 90, 50).view(np.uint32).tobytes()
    )


class _Strings:
    def __init__(self, producer, sentences):
        self.p, self.strings = producer, [s.encode() for s in sentences]

    def __arrow_c_stream__(self, requested_schema=None):
        p = self.p
        return p.stream(p.schema(b"u"), [p.strings(b"u", self.strings)])


def test_one_nogil_section_per_call():
    pipeline = _pipeline(fasttext=True)
    text = _text(texts(200))
    before = _native._test_nogil_sections()
    _fused(pipeline, text, 0, 200, 200)
    assert _native._test_nogil_sections() - before == 1
    before = _native._test_nogil_sections()
    _sequence(pipeline, text, 0, 200, 200)
    assert _native._test_nogil_sections() - before == 3  # encode, pool, normalize


def _errors_pipeline():
    words = ["ok", "tiny", "huge"]
    rows = np.array([[1, 2, 3, 4], [1e-30] * 4, [3e38] * 4], F32)
    return _pipeline(wv=WordVectors.from_arrays(rows, words=words))


@pytest.mark.parametrize(
    "sentences",
    [
        ["ok", "huge huge", "ok"],  # pooling
        ["ok", "tiny", "ok"],  # normalize: cannot be normalized
        ["ok", "tiny", "huge huge"],  # pooling over the whole range comes first
        ["ok", "\udcff", "ok"],  # not this route: a lone surrogate never reaches a buffer
    ],
)
def test_errors_and_their_order_are_those_of_the_calls(sentences):
    pipeline = _errors_pipeline()
    try:
        text = _text(sentences)
    except UnicodeEncodeError:
        return
    got = _same(pipeline, text, 0, 3, 3, first_row=50)
    assert got[0] is ValueError and ("row 51" in got[1] or "row 52" in got[1])


def test_invalid_utf8_and_too_long_sentences_are_stage_1():
    pipeline = _errors_pipeline()
    data = np.frombuffer(b"ok\nhuge huge\nok \xff\n", np.uint8)
    got = _same(pipeline, (data, None), 0, len(data), 3, first_row=7)
    assert got[0] is UnicodeDecodeError and "in sentence 9" in got[1]
    previous = _native._test_sentence_limit(4)
    try:
        got = _same(pipeline, _text(["ok", "huge huge", "ok"]), 0, 3, 3, out=np.zeros((3, 3), F32))
    finally:
        _native._test_sentence_limit(previous)
    assert got[0] is ValueError and "sentence 1 has 9 bytes" in got[1]  # before out's error


def _read_only():
    out = np.zeros((3, 4), F32)
    out.flags.writeable = False
    return out


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.zeros((3, 4), np.float64),
        lambda: np.zeros((2, 4), F32),
        lambda: np.zeros((3, 8), F32)[:, ::2],
        lambda: np.zeros((3, 4), F32, order="F"),
        _read_only,
        lambda: [[0.0] * 4] * 3,
    ],
)
def test_out_is_checked_as_check_out_does(make):
    got = _same(_errors_pipeline(), _text(["ok"] * 3), 0, 3, 3, out=make())
    assert got[0] is ValueError and got[1].startswith("out must be a writable")


def test_out_must_not_overlap_the_word_rows_or_weights():
    pipeline = _pipeline(
        wv=WordVectors.from_arrays(np.ones((5, 4), F32), words=[f"w{i}" for i in range(5)])
    )
    vectors, weights = np.ones((5, 4), F32), np.ones(20, F32)
    kernel = _native.WeightedMean(vectors, weights[:5])
    text = _text(["w1"] * 5)
    batch = pipeline.lookup.encode(text[0], text[1], 0, 5, 0, new_workspace())
    for out in (vectors, weights.reshape(5, 4)):
        with pytest.raises(ValueError, match="must not share memory with the vectors") as old:
            kernel.pool(batch, out)
        with pytest.raises(ValueError) as new:
            _native.pool_text(
                kernel, pipeline.lookup, new_workspace(), text, 0, 5, 5, 0, out, False
            )
        assert str(new.value) == str(old.value)
    out = np.zeros((5, 4), F32)
    _native.pool_text(kernel, pipeline.lookup, new_workspace(), text, 0, 5, 5, 0, out, False)
    np.testing.assert_array_equal(out, kernel.pool(batch))


def test_a_wrong_sentence_count_is_a_changed_source():
    pipeline = _pipeline()
    data = np.frombuffer(b"w1\nw2\nw3\n", np.uint8)
    got = _same(pipeline, (data, None), 0, 9, 2)
    assert got == (RuntimeError, "the source changed since it was scanned")


def test_after_encode_runs_before_the_errors_of_later_stages():
    pipeline = _errors_pipeline()
    calls = []

    def changed():
        calls.append(1)
        raise RuntimeError("a.txt changed while it was read")

    got = _same(pipeline, _text(["ok", "huge huge"]), 0, 2, 2, after=changed)
    assert got == (RuntimeError, "a.txt changed while it was read")
    data = np.frombuffer(b"ok\n\xff\n", np.uint8)
    got = _same(pipeline, (data, None), 0, 5, 2, after=changed)
    assert got[0] is UnicodeDecodeError  # stage 1 before the source's own check
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("first_row", "kind"),
    [
        (True, TypeError),
        ("3", TypeError),
        (-1, ValueError),
        (2**63, ValueError),
        (2**63 - 2, ValueError),
    ],
)
def test_first_row_is_checked_as_batch_does(first_row, kind):
    pipeline = _pipeline()
    got = _same(pipeline, _text(["w1", "w2", "w3"]), 0, 3, 3, first_row=first_row)
    assert got[0] is kind


# The corruptions of _test_corrupt, and what they do to the arrays of the Batch route.
def _corrupt(kind, tokens, offsets, ngram_offsets, ngrams):
    if kind == 1:
        offsets[0] = 1
    elif kind == 2:
        offsets[1] = offsets[-1] + 1
    elif kind == 3:
        offsets[-1] += 1
    elif kind == 4:
        tokens[0] = _N_WORDS
    elif kind == 5:
        ngrams[0] = _N_BUCKETS
    elif kind == 6:
        ngram_offsets = ngram_offsets[:-1]
    elif kind == 7:
        tokens[0] = OOV_TOKEN
    elif kind == 8:
        offsets = np.append(offsets, offsets[-1])
    elif kind == 9:
        ngram_offsets[0] = 1
    elif kind == 10:
        ngram_offsets[-1] += 1
    return tokens, offsets, ngram_offsets, ngrams


_N_WORDS = 200
_N_BUCKETS = 97


@pytest.mark.parametrize("kind", range(1, 11))
@pytest.mark.parametrize("fasttext", [False, True])
def test_every_check_of_the_batch_still_fires(kind, fasttext):
    from fse._contract import Batch

    pipeline = _pipeline(fasttext=fasttext, normalize=False)
    sentences = ["w1 zzz w2", "w3 zzy", "w4"]
    text = _text(sentences)
    ws = new_workspace()
    batch = pipeline.lookup.encode(text[0], text[1], 0, 3, 0, ws)
    arrays = [
        np.array(a) if a is not None else None
        for a in (batch.tokens, batch.offsets, batch.ngram_offsets, batch.ngrams)
    ]
    if (kind in (5, 6, 9, 10)) and not fasttext:
        return  # no ngram buffers to break
    tokens, offsets, ngram_offsets, ngrams = _corrupt(kind, *arrays)
    try:
        rows = len(offsets) - 1
        if rows != 3:
            raise RuntimeError("the source changed since it was scanned")
        old = Batch(0, offsets, tokens, ngram_offsets, ngrams)
        pipeline.backend.rows(old, pipeline.prepared, np.zeros((3, pipeline.output_dim), F32))
        want = None
    except Exception as error:  # noqa: BLE001 - compared by type and message
        want = (type(error), str(error))
    previous = _native._test_corrupt(kind)
    try:
        got = _outcome(_fused, pipeline, text, 0, 3, 3)
    finally:
        _native._test_corrupt(previous)
    assert want is not None
    assert got == want


def test_a_lookup_that_was_never_built_and_a_busy_workspace_raise():
    pipeline = _pipeline()
    lookup = _native.TextLookup.__new__(_native.TextLookup)
    out = np.zeros((1, pipeline.output_dim), F32)
    text = _text(["w1"])
    kernel = pipeline.prepared.state.kernel
    with pytest.raises(RuntimeError, match="was not built"):
        _native.pool_text(kernel, lookup, new_workspace(), text, 0, 1, 1, 0, out, True)
    with pytest.raises(TypeError, match="text must be"):
        _native.pool_text(kernel, pipeline.lookup, new_workspace(), [text], 0, 1, 1, 0, out, True)


def test_the_floating_point_environment_of_the_thread_changes_no_bit():
    pipeline = _pipeline()
    text = _text(texts(200))
    want = _fused(pipeline, text, 0, 200, 200).view(np.uint32)
    saved = _native._test_fpenv_set(flush=True, round_up=True)
    try:
        inside = _native._test_fpenv_get()
        got = _fused(pipeline, text, 0, 200, 200).view(np.uint32)
        assert _native._test_fpenv_get() == inside  # restored after the call
        with pytest.raises(ValueError):
            _fused(_errors_pipeline(), _text(["huge huge"]), 0, 1, 1)
        assert _native._test_fpenv_get() == inside  # and after an error
    finally:
        _native._test_fpenv_restore(saved)
    np.testing.assert_array_equal(got, want)
