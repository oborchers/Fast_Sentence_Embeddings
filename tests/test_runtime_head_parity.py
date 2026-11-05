"""A lone head runs through the one runtime of heads and must be the lone head of 2.0: the
frozen pipeline (tests/support/head_legacy.py) gives the same bits, vectors, sinks, errors
and error rows on every backend and path."""

import numpy as np
import pytest

from fse._contract import Link
from fse._runtime import HeadSpec, MemorySink, make_head
from fse._text import StringSource
from support.head_cases import BACKENDS, case_spec, head_of, sentences
from support.head_legacy import legacy_make_head
from support.heads import bits, own_of
from support.heads_cases import column_head, overflow_spec

F32 = np.float32
TEXTS = sentences(60)


def _pair(spec, head, backend, isa, vectors):
    old = legacy_make_head(
        spec, head, backend, own=own_of(spec, backend, isa), with_vectors=vectors, isa=isa
    )
    new = make_head(
        spec, head, backend, own=own_of(spec, backend, isa), with_vectors=vectors, isa=isa
    )
    return old, new


def _same(a, b):
    if a is None or b is None:
        assert a is b
    elif hasattr(a, "codes"):
        np.testing.assert_array_equal(a.codes, b.codes)
        np.testing.assert_array_equal(bits(a.scales), bits(b.scales))
    elif a.dtype.kind == "f":
        np.testing.assert_array_equal(bits(a), bits(b))
    else:
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("k", [1, 5])
def test_a_lone_head_has_the_bits_of_2_0(backend, dtype, vectors, normalized, normalize, k):
    name, isa = backend
    spec = case_spec(dtype, fasttext=True, normalize=normalize)
    head = head_of(spec, k, normalized)
    old, new = _pair(spec, head, name, isa, vectors)
    for output in ("logits", "proba", "labels"):
        for vdtype in ("float32", "int8") if vectors else ("float32",):
            a = old.infer(StringSource(TEXTS), output=output, workers=2, vector_dtype=vdtype)
            b = new.infer(StringSource(TEXTS), output=output, workers=2, vector_dtype=vdtype)
            _same(a[0], b[0])
            _same(a[1], b[1])


@pytest.mark.parametrize("vectors", [False, True])
def test_the_label_rule_linear2_and_multilabel_are_those_of_2_0(vectors):
    from fse._contract import Head

    spec = case_spec("float32", fasttext=False)
    for link, rule in ((Link.MULTILABEL, Head.LINEAR2), (Link.SOFTMAX, Head.LINEAR2)):
        base = head_of(spec, 4, False, link=link)
        limits = np.linspace(0.2, 0.8, 4).astype(F32) if link is Link.MULTILABEL else None
        head = HeadSpec(base.weight, base.bias, False, link, limits, rule)
        old, new = _pair(spec, head, "cpu", None, vectors)
        for output in ("proba", "labels"):
            a = old.infer(StringSource(TEXTS), output=output, workers=3)
            b = new.infer(StringSource(TEXTS), output=output, workers=3)
            _same(a[0], b[0])


def test_sinks_and_first_row_are_those_of_2_0():
    spec = case_spec("float32", fasttext=False)
    head = head_of(spec, 3, False)
    old, new = _pair(spec, head, "cpu", None, True)
    sinks = []
    for pipeline in (old, new):
        out, vec = MemorySink(3), MemorySink(spec.output_dim)
        got = pipeline.infer(
            StringSource(TEXTS), output="logits", sink=out, vector_sink=vec, first_row=0, workers=2
        )
        assert got == (None, None)
        sinks.append((out.result(), vec.result()))
    _same(sinks[0][0], sinks[1][0])
    _same(sinks[0][1], sinks[1][1])


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("first_row", [0, 5])
def test_errors_and_their_rows_are_those_of_2_0(vectors, workers, first_row):
    spec = overflow_spec()
    cases = [
        (column_head(spec, 0, F32(3e38)), ["w1", "w5 w6", "w1"]),  # pooling
        (column_head(spec, 1, F32(3e38), bias=F32(3e38)), ["w1", "w1", "w7"]),  # logits
        (column_head(spec, 0, 1.0, normalized=True), ["w1", "w1"]),  # fine
    ]
    for head, texts in cases:
        old, new = _pair(spec, head, "cpu", None, vectors)
        outcomes = []
        for pipeline in (old, new):
            try:
                result = pipeline.infer(
                    StringSource(texts), output="logits", workers=workers, first_row=first_row
                )
                outcomes.append(("ok", bits(result[0]).tobytes()))
            except ValueError as error:
                outcomes.append(("error", str(error)))
        assert outcomes[0] == outcomes[1]


def test_the_checks_and_their_messages_are_those_of_2_0():
    spec = case_spec("float32", fasttext=False)
    head = head_of(spec, 2, False)
    old, new = _pair(spec, head, "cpu", None, False)
    for kwargs in (
        {"output": "values"},
        {"output": "labels", "sink": MemorySink(1)},
        {"output": "logits", "sink": MemorySink(3)},
        {"output": "logits", "vector_sink": MemorySink(spec.output_dim)},
        {"output": "logits", "workers": 0},
        {"output": "logits", "vector_dtype": "int4"},
    ):
        messages = []
        for pipeline in (old, new):
            with pytest.raises(ValueError, match=r".") as caught:
                pipeline.infer(StringSource(["w1"]), **kwargs)
            messages.append(str(caught.value))
        assert messages[0] == messages[1]
