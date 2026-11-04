"""Independent multilabel probabilities and decisions on every supported CPU backend."""

import numpy as np
import pytest

from fse import _native
from fse import _reference_head as ref
from fse._backends import get_backend
from fse._contract import Link
from fse._runtime import Pipeline
from fse._runtime._head import HeadSpec, make_head
from fse._text import StringSource, make_vocabulary
from support import head as support
from support.head_cases import BACKENDS, case_spec, head_of, sentences


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda item: item[1] or item[0])
@pytest.mark.parametrize("k", [1, 2, 17, 33])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("path", ["folded", "normalized", "vectors", "standalone"])
def test_every_path_matches_reference(backend, k, dtype, path):
    name, isa = backend
    spec = case_spec(dtype, fasttext=True, normalize=path == "normalized")
    base = head_of(spec, k, path == "normalized")
    limits = np.resize(np.array([0, 0.5, 0.2, 1], np.float32), k)
    head = HeadSpec(base.weight, base.bias, base.normalized, Link.MULTILABEL, limits)
    own = Pipeline(spec, name, isa=isa, vocabulary=make_vocabulary(spec.word_vectors))
    pipeline = make_head(
        spec, head, name, own=own, with_vectors=path == "vectors", isa=isa, workers=1
    )
    if path == "standalone":
        head = HeadSpec(None, head.bias, False, Link.MULTILABEL, limits)
        pipeline = type(pipeline)(pipeline.reader, head, folded=True)
    texts = sentences(16)
    z = support.logits(
        spec,
        base.weight,
        base.bias,
        texts,
        normalized=base.normalized,
        folded=path in ("folded", "standalone"),
    )
    for output in ("logits", "proba", "labels"):
        got, _ = pipeline.infer(StringSource(texts), output=output, workers=2, chunk_bytes=8192)
        if output == "labels":
            assert got.dtype == np.bool_
            np.testing.assert_array_equal(got, ref.multilabel(z, limits))
        else:
            want = z if output == "logits" else ref.sigmoid(z)
            np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda item: item[1] or item[0])
@pytest.mark.parametrize("k", [1, 3, 4, 5, 8, 9, 16, 17])
def test_link_tails_subnormals_saturation_and_empty(backend, k):
    name, isa = backend
    host = get_backend(name, isa=isa)
    z = np.resize(np.array([-104, -100, -88, -0.0, 0.0, 0.1, 20, 104], np.float32), (3, k))
    got = np.empty_like(z)
    host.outputs(z, "proba", Link.MULTILABEL, got)
    np.testing.assert_array_equal(got.view(np.uint32), ref.sigmoid(z).view(np.uint32))
    assert (got.sum(axis=1) != 1).any()  # never normalize independent labels
    empty = np.empty((0, k), np.float32)
    host.outputs(empty, "proba", Link.MULTILABEL, empty.copy())


@pytest.mark.parametrize("isa", _native.available_isas())
def test_native_multilabel_restores_the_float_environment(isa):
    z = np.array([[-100, -95, 0.1, -0.3, -88, -103.5, 0.7, -1e-30, 95]], np.float32)
    want = ref.sigmoid(z)
    out = np.empty_like(z)
    saved = _native._test_fpenv_set(flush=True, round_up=True)
    try:
        before = _native._test_fpenv_get()
        _native.head_outputs(z, "proba", Link.MULTILABEL, out, isa=isa)
        after = _native._test_fpenv_get()
    finally:
        _native._test_fpenv_restore(saved)
    assert before == after
    np.testing.assert_array_equal(out.view(np.uint32), want.view(np.uint32))


def test_cases_distinguish_normalized_ovr_and_wrong_threshold_comparison():
    z = np.zeros((2, 3), np.float32)
    limits = np.full(3, 0.5, np.float32)
    assert not np.array_equal(ref.sigmoid(z).view(np.uint32), ref.ovr(z).view(np.uint32))
    assert not np.array_equal(ref.multilabel(z, limits), ref.sigmoid(z) >= limits)
