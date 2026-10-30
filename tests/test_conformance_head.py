"""Conformance of linear heads: every CPU backend against fse._reference_head, bit for bit,
on every path (folded, normalized, unfolded with the vectors, standalone), for float32,
float16 and int8 tables, FastText with unknown words, empty, one-token and long sentences,
k = 1, 2 and 17, ties, and the errors of the fold and of normalize@1. The meta-tests in
tests/test_conformance_head_meta.py check that these cases catch the usual mistakes."""

import numpy as np
import pytest

from fse import _native
from fse import _reference_head as ref
from fse._backends import UnsupportedByBackendError, get_backend
from fse._contract import Link
from fse._runtime import Pipeline
from fse._runtime._head import HeadPipeline, HeadSpec, fold, make_head
from fse._stream import StreamSource
from fse._text import StringSource, TokenListSource, make_vocabulary
from support import head as support
from support.head_cases import BACKENDS, LINKS, case_spec, head_of, sentences

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _own(spec, backend, isa):
    return Pipeline(spec, backend, isa=isa, vocabulary=make_vocabulary(spec.word_vectors))


def _source(kind, texts):
    if kind == "tokens":
        return TokenListSource([t.split(" ") if t else [] for t in texts])
    if kind == "stream":
        return StreamSource(iter(texts), chunk_rows=13)
    return StringSource(texts)


def _check(got, z, output, link):
    want = support.outputs(z, output, link)
    if output == "labels":
        assert got.tolist() == want.tolist()
    else:
        np.testing.assert_array_equal(bits(got), bits(want))


CASES = [
    (dtype, fasttext, k)
    for dtype in ("float32", "float16", "int8")
    for fasttext in (False, True)
    for k in (1, 2, 17)
]


@pytest.mark.parametrize(("dtype", "fasttext", "k"), CASES)
@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize("normalized", [False, True], ids=["raw", "normalized"])
def test_every_backend_on_every_path(backend, dtype, fasttext, k, normalized):
    name, isa = backend
    spec = case_spec(dtype, fasttext, normalize=normalized)
    head = head_of(spec, k, normalized)
    texts = sentences()
    pipeline = make_head(spec, head, name, own=_own(spec, name, isa), isa=isa, workers=2)
    z = support.logits(spec, head.weight, head.bias, texts, normalized=normalized)
    for output in ("logits", "proba", "labels"):
        got, _ = pipeline.infer(StringSource(texts), output=output, workers=3, chunk_bytes=1 << 16)
        _check(got, z, output, head.link)


@pytest.mark.parametrize(
    ("dtype", "fasttext"), [("float32", True), ("int8", False), ("float16", True)]
)
@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize(("normalize", "normalized"), [(True, True), (False, False), (True, False)])
def test_the_unfolded_path_with_vectors(backend, dtype, fasttext, normalize, normalized):
    name, isa = backend
    spec = case_spec(dtype, fasttext, normalize=normalize)
    head = head_of(spec, 17, normalized)
    own = _own(spec, name, isa)
    pipeline = make_head(spec, head, name, own=own, with_vectors=True, isa=isa)
    texts = sentences()
    got, vectors = pipeline.infer(StringSource(texts), output="logits", workers=2)
    z = support.logits(spec, head.weight, head.bias, texts, normalized=normalized, folded=False)
    np.testing.assert_array_equal(bits(got), bits(z))
    np.testing.assert_array_equal(bits(vectors), bits(support.pooled(spec, texts, spec.steps)))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_a_standalone_table_predicts_the_bits_of_the_complete_head(backend, dtype):
    name, isa = backend
    spec = case_spec(dtype, fasttext=True, normalize=False)
    head = head_of(spec, 2, False)
    pool = fold(get_backend(name, isa=isa), spec.pool_input(), head.weight, 1)
    table = Pipeline(spec, name, isa=isa, pool=pool, steps=())
    standalone = HeadPipeline(table, HeadSpec(None, head.bias, False, head.link), folded=True)
    texts = sentences()
    got, _ = standalone.infer(StringSource(texts), output="proba", workers=2)
    z = support.logits(spec, head.weight, head.bias, texts)
    _check(got, z, "proba", head.link)


@pytest.mark.parametrize("kind", ["strings", "tokens", "stream"])
@pytest.mark.parametrize("workers", [1, 4])
def test_worker_counts_and_input_classes_give_the_same_bits(kind, workers):
    spec = case_spec("float16", fasttext=True, normalize=False, tokenizer="external")
    head = head_of(spec, 17, False)
    pipeline = make_head(spec, head, "cpu", own=_own(spec, "cpu", None))
    texts = sentences()
    got, _ = pipeline.infer(
        _source(kind, texts), output="proba", workers=workers, chunk_bytes=1 << 16
    )
    _check(got, support.logits(spec, head.weight, head.bias, texts), "proba", head.link)


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
def test_every_link_and_ties(backend, link):
    name, isa = backend
    spec = case_spec("float32", fasttext=False, normalize=False)
    k = 1 if link is Link.SIGMOID else 17
    head = head_of(spec, k, False, link=link)
    if k > 1:  # two equal outputs tie on every row: the smaller index wins
        weight, bias = head.weight.copy(), head.bias.copy()
        weight[5] = weight[3]
        bias[3] = bias[5] = 50.0  # the two largest logits of every row
        head = HeadSpec(weight, bias, False, link)
    pipeline = make_head(spec, head, name, own=_own(spec, name, isa), isa=isa)
    texts = sentences()
    z = support.logits(spec, head.weight, head.bias, texts)
    for output in ("proba", "labels"):
        got, _ = pipeline.infer(StringSource(texts), output=output, workers=2)
        _check(got, z, output, link)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
def test_a_folded_value_that_overflows_raises_at_prepare(backend):
    name, isa = backend
    spec = case_spec("float32", fasttext=False, normalize=False)
    weight = np.full((2, spec.output_dim), 3e38, F32)
    head = HeadSpec(weight, np.zeros(2, F32), False, Link.SOFTMAX)
    with pytest.raises(ValueError, match="does not fold into the table"):
        make_head(spec, head, name, own=_own(spec, name, isa), isa=isa)
    with pytest.raises(ValueError, match="overflows float32"):
        ref.fold(spec.pool_input().vectors, None, weight)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b[1] or b[0])
def test_a_row_whose_norm_overflows_raises_on_the_normalized_path(backend):
    name, isa = backend
    spec = case_spec("float32", fasttext=False, normalize=False, huge=True)
    head = head_of(spec, 2, True)
    pipeline = make_head(spec, head, name, own=_own(spec, name, isa), isa=isa)
    texts = ["w1 w2", "w0 w0", "w3"]  # w0 is huge: the sum of squares of row 1 overflows
    with pytest.raises(ValueError, match="row 1 cannot be normalized"):
        pipeline.infer(StringSource(texts), output="labels", workers=1)
    raw = make_head(spec, head_of(spec, 2, False), name, own=_own(spec, name, isa), isa=isa)
    raw.infer(StringSource(texts), output="labels", workers=1)  # the raw head folds and runs


def test_the_mlx_backend_refuses_heads():
    try:
        backend = get_backend("mlx")
    except UnsupportedByBackendError:
        pytest.skip("mlx cannot run here")
    with pytest.raises(UnsupportedByBackendError, match="mlx backend cannot run linear heads"):
        backend.capabilities().check_heads("mlx")


def test_isas_cover_this_machine():
    assert {isa for name, isa in BACKENDS if name == "cpu"} == set(_native.available_isas())


def _float64_proba(z, link):
    if link is Link.SIGMOID:
        e = np.exp(-np.abs(z[:, 0]))
        p = np.where(z[:, 0] >= 0, 1 / (1 + e), e / (1 + e))
        return np.stack([1 - p, p], axis=1)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("dtype", "fasttext"), [("float32", True), ("float16", False), ("int8", True)]
)
@pytest.mark.parametrize("k", [1, 17])
def test_folded_and_unfolded_logits_agree_within_the_bound(dtype, fasttext, k):
    """Folding changes the order of the roundings only: both paths are within their bound
    of the exact head on fse's vectors, labels differ only where that allows it, and so
    do probabilities."""
    from support import head_bound

    spec = case_spec(dtype, fasttext, normalize=False)
    head = head_of(spec, k, False)
    texts = sentences(400, seed=5)
    zf = support.logits(spec, head.weight, head.bias, texts)
    zu = support.logits(spec, head.weight, head.bias, texts, folded=False)
    r = support.pooled(spec, texts)
    a64, b64 = head.weight.astype(np.float64), head.bias.astype(np.float64)
    z_star = r.astype(np.float64) @ a64.T + b64
    batch = support.reference_batch(spec, texts)
    table = support.folded_pool(spec.pool_input(), head.weight)

    def folded_rows(kind, i):
        return table[0][i] if kind == "word" else table[3][i]

    bound_u = head_bound.unfolded(r, a64, b64, head.weight, head.bias, zu)
    bound_f = head_bound.folded(spec, batch, folded_rows, r, a64, b64, head.weight, head.bias, zf)
    assert (np.abs(zu - z_star) <= bound_u).all()
    assert (np.abs(zf - z_star) <= bound_f).all()
    changed = (
        ref.labels(zf) != np.argmax(z_star, axis=1)
        if k > 1
        else ref.labels(zf) != (z_star[:, 0] > 0)
    )
    assert not (changed & ~head_bound.allowed_label_changes(z_star, bound_f)).any()
    tolerance = head_bound.proba_tolerance(head.link.id, bound_f, k)
    difference = np.abs(ref.proba(zf, head.link) - _float64_proba(z_star, head.link)).max(axis=1)
    assert (difference <= tolerance).all()
    assert bits(zf).tolist() != bits(zu).tolist()  # the paths differ in their last bits
