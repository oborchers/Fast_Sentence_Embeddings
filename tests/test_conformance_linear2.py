"""linear@2 for single-label heads: the first class of largest probability, on every CPU
backend, bit for bit as the reference decides it."""

import numpy as np
import pytest

from fse import _reference_head as ref
from fse._backends import get_backend
from fse._contract import Head, Link
from fse._runtime import Pipeline
from fse._runtime._head import HeadSpec, make_head
from fse._text import StringSource, make_vocabulary
from support import head as support
from support.head_cases import BACKENDS, case_spec, head_of, sentences

# (link, k): the single-label heads with probabilities, with the tails of the columns.
HEADS = [
    (Link.SIGMOID, 1),
    (Link.SOFTMAX, 1),
    (Link.SOFTMAX, 2),
    (Link.SOFTMAX, 3),
    (Link.SOFTMAX, 5),
    (Link.SOFTMAX, 17),
    (Link.OVR, 2),
    (Link.OVR, 3),
    (Link.OVR, 5),
    (Link.OVR, 17),
]


def _head(spec, k, normalized, link):
    base = head_of(spec, k, normalized, link)
    return HeadSpec(base.weight, base.bias, normalized, link, rule=Head.LINEAR2)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda item: item[1] or item[0])
@pytest.mark.parametrize(("link", "k"), HEADS, ids=lambda item: str(item))
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("path", ["folded", "normalized", "vectors", "standalone"])
def test_every_path_is_the_argmax_of_the_probabilities(backend, link, k, dtype, path):
    name, isa = backend
    spec = case_spec(dtype, fasttext=True, normalize=path == "normalized")
    head = _head(spec, k, path == "normalized", link)
    own = Pipeline(spec, name, isa=isa, vocabulary=make_vocabulary(spec.word_vectors))
    pipeline = make_head(
        spec, head, name, own=own, with_vectors=path == "vectors", isa=isa, workers=1
    )
    if path == "standalone":
        head = HeadSpec(None, head.bias, False, link, rule=Head.LINEAR2)
        pipeline = type(pipeline)(pipeline.reader, head, folded=True)
    texts = sentences(16)
    z = support.logits(
        spec,
        head_of(spec, k, path == "normalized", link).weight,
        head.bias,
        texts,
        normalized=path == "normalized",
        folded=path in ("folded", "standalone"),
    )
    want = ref.predicted(z, link)
    got, _ = pipeline.infer(StringSource(texts), output="labels", workers=2, chunk_bytes=8192)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, want)
    proba, _ = pipeline.infer(StringSource(texts), output="proba", workers=2, chunk_bytes=8192)
    np.testing.assert_array_equal(got, np.argmax(proba, axis=1))
    empty, _ = pipeline.infer(StringSource([]), output="labels", workers=1)
    assert empty.shape == (0,)
    assert empty.dtype == np.int64


# Rows at the limits of float32: complete underflow, partial underflow, saturation, a tiny
# positive logit whose probability is exactly one half.
LIMITS = [
    (Link.OVR, [-107, -106, -105], 0),  # all exp@1 are 0: uniform, the first class
    (Link.OVR, [-90, -10, -95], 1),
    (Link.OVR, [30, 40, 0], 0),  # sigmoid is 1.0 twice: the first of the largest
    (Link.SIGMOID, [2.0**-30], 0),  # p = 0.5, the first of the equal [0.5, 0.5]
    (Link.SOFTMAX, [3, 3, 3], 0),
]


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda item: item[1] or item[0])
@pytest.mark.parametrize(("link", "row", "label"), LIMITS, ids=lambda item: str(item)[:24])
def test_limits_decide_by_the_probabilities_not_the_logits(backend, link, row, label):
    name, isa = backend
    host = get_backend(name, isa=isa)
    z = np.array([row], np.float32)
    assert ref.predicted(z, link).tolist() == [label]
    probabilities = np.empty((1, 2 if len(row) == 1 else len(row)), np.float32)
    host.outputs(z, "proba", link, probabilities)
    np.testing.assert_array_equal(probabilities.view(np.uint32), ref.proba(z, link).view(np.uint32))
    assert int(np.argmax(probabilities, axis=1)[0]) == label


def test_linear_one_decides_other_than_linear_two_where_probabilities_tie():
    z = np.array([[30, 40, 0]], np.float32)  # ovr: sigmoid(30) = sigmoid(40) = 1.0
    assert ref.labels(z).tolist() == [1]
    assert ref.predicted(z, Link.OVR).tolist() == [0]
    tiny = np.array([[2.0**-30]], np.float32)
    assert ref.labels(tiny).tolist() == [1]
    assert ref.predicted(tiny, Link.SIGMOID).tolist() == [0]
