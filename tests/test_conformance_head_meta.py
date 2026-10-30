"""Meta-tests of the head conformance suite: each usual mistake of an implementation,
written as a numpy model, gives other bits (or no error) than the reference on the cases
of tests/test_conformance_head.py, so a backend that makes it cannot pass."""

import numpy as np
import pytest

from fse import _reference
from fse import _reference_head as ref
from fse._contract import Link, Normalize
from support import head as support
from support.head_cases import case_spec, head_of, sentences

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _folded_parts(spec, head, texts):
    batch = support.reference_batch(spec, texts)
    args = support.folded_pool(spec.pool_input(), head.weight)
    return batch, args


def _with_bias_in_the_table(spec, head, texts):
    """Folds b into every row, so b is added per token and an empty sentence gives 0."""
    batch, args = _folded_parts(spec, head, texts)
    rows = (args[0] + head.bias).astype(F32)
    ngrams = None if args[3] is None else (args[3] + head.bias).astype(F32)
    return _reference.pool_weighted_mean(batch, rows, args[1], None, ngrams, None, args[5])


def _bias_before_the_mean(spec, head, texts):
    batch, args = _folded_parts(spec, head, texts)
    accs, counts = _reference.accumulate(batch, *args)
    out = np.zeros_like(accs)
    for s, (acc, n) in enumerate(zip(accs, counts, strict=True)):
        out[s] = (F32(1) / F32(max(n, 1))) * (acc + head.bias) if n else head.bias
    return out.astype(F32)


def _bias_twice(spec, head, texts):
    z = support.logits(spec, head.weight, head.bias, texts)
    return (z + head.bias).astype(F32)


def _no_norm(spec, head, texts):
    return support.logits(spec, head.weight, head.bias, texts, normalized=False, folded=False)


MISTAKES = {
    "folds b into the table": _with_bias_in_the_table,
    "adds b before 1/n": _bias_before_the_mean,
    "adds b twice": _bias_twice,
}


@pytest.mark.parametrize("mistake", MISTAKES)
@pytest.mark.parametrize("k", [1, 17])
def test_the_cases_catch_a_wrong_bias(mistake, k):
    spec = case_spec("float32", fasttext=True, normalize=False)
    head = head_of(spec, k, False)
    texts = sentences()
    want = support.logits(spec, head.weight, head.bias, texts)
    got = MISTAKES[mistake](spec, head, texts)
    assert bits(got).tolist() != bits(want).tolist()


def test_the_cases_catch_a_skipped_norm():
    spec = case_spec("int8", fasttext=True, normalize=True)
    head = head_of(spec, 2, True)
    texts = sentences()
    want = support.logits(spec, head.weight, head.bias, texts, normalized=True)
    assert bits(_no_norm(spec, head, texts)).tolist() != bits(want).tolist()


def test_the_cases_catch_skipped_errors_of_the_norm():
    """The reference raises on the huge case; a backend that skips the check of
    normalize@1 (or the norm) gives finite logits instead."""
    spec = case_spec("float32", fasttext=False, normalize=False, huge=True)
    head = head_of(spec, 2, True)
    texts = ["w1 w2", "w0 w0", "w3"]
    with pytest.raises(ValueError, match="row 1 cannot be normalized"):
        support.logits(spec, head.weight, head.bias, texts, normalized=True)
    rows = support.pooled(spec, texts)
    with np.errstate(over="ignore", invalid="ignore"):
        scale = F32(1) / np.sqrt(_reference.dot16(rows, rows))  # no check: inf -> 0
    unchecked = rows * scale[:, None]
    assert np.isfinite(ref.logits(unchecked, head.weight, head.bias)).all()


def test_the_cases_catch_ties_to_the_larger_index():
    spec = case_spec("float32", fasttext=False, normalize=False)
    head = head_of(spec, 17, False)
    weight, bias = head.weight.copy(), head.bias.copy()
    weight[5] = weight[3]
    bias[3] = bias[5] = 50.0  # the two largest logits of every row
    z = support.logits(spec, weight, bias, sentences())
    want = ref.labels(z)
    last = z.shape[1] - 1 - np.argmax(z[:, ::-1], axis=1)
    assert want.tolist() != last.tolist()


def test_the_cases_catch_the_naive_sigmoid():
    spec = case_spec("float32", fasttext=True, normalize=False)
    head = head_of(spec, 1, False)
    z = support.logits(spec, head.weight, head.bias, sentences())[:, 0]
    assert (z < 0).any()
    with np.errstate(over="ignore"):
        naive = (F32(1) / (F32(1) + np.exp(-z, dtype=F32))).astype(F32)
    assert bits(naive).tolist() != bits(ref.sigmoid(z)).tolist()


def test_the_cases_catch_another_order_of_the_softmax_sum():
    spec = case_spec("float16", fasttext=True, normalize=False)
    head = head_of(spec, 17, False)
    z = support.logits(spec, head.weight, head.bias, sentences())
    e = ref.exp((z - z.max(axis=1, keepdims=True)).astype(F32))
    backwards = np.zeros(len(e), F32)
    for j in reversed(range(e.shape[1])):
        backwards = (backwards + e[:, j]).astype(F32)
    other = (e / backwards[:, None]).astype(F32)
    assert bits(other).tolist() != bits(ref.proba(z, Link.SOFTMAX)).tolist()


def test_the_normalized_case_differs_from_the_raw_one():
    """A backend that normalizes where the head reads raw means is caught too."""
    spec = case_spec("float32", fasttext=False, normalize=False)
    head = head_of(spec, 2, False)
    texts = sentences()
    raw = support.logits(spec, head.weight, head.bias, texts)
    steps = _reference.postprocess(support.pooled(spec, texts), [Normalize()])
    assert bits(ref.logits(steps, head.weight, head.bias)).tolist() != bits(raw).tolist()
