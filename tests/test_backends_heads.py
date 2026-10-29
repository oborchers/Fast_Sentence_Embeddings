"""The head operations of the backends (logits, outputs) against fse._reference_head."""

import numpy as np
import pytest

from fse import _native
from fse import _reference_head as ref
from fse._backends import UnsupportedByBackendError, get_backend
from fse._contract import Link

F32 = np.float32
BACKENDS = [*(("cpu", isa) for isa in _native.available_isas()), ("numpy", None)]


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _backend(spec):
    name, isa = spec
    return get_backend(name, isa=isa)


@pytest.fixture(params=BACKENDS, ids=lambda b: b[1] or b[0])
def backend(request):
    return _backend(request.param)


def test_cpu_and_numpy_run_heads_and_mlx_says_it_cannot():
    for spec in BACKENDS:
        assert _backend(spec).capabilities().heads
    try:
        mlx = get_backend("mlx")
    except UnsupportedByBackendError:
        pytest.skip("mlx cannot run here")
    with pytest.raises(UnsupportedByBackendError, match="mlx backend cannot run linear heads"):
        mlx.capabilities().check_heads("mlx")


def test_logits_are_the_reference(backend):
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((30, 21)).astype(F32)
    weights = rng.standard_normal((3, 21)).astype(F32)
    bias = np.array([1.0, -0.0, 0.25], F32)
    out = np.empty((30, 3), F32)
    backend.logits(rows, weights, bias, out)
    np.testing.assert_array_equal(bits(out), bits(ref.logits(rows, weights, bias)))
    folded = rng.standard_normal((30, 3)).astype(F32)
    want = ref.logits(folded, None, bias)
    backend.logits(folded, None, bias, folded)  # in place
    np.testing.assert_array_equal(bits(folded), bits(want))


def test_logits_name_the_row_that_is_not_finite(backend):
    rows = np.array([[0.0], [3e38]], F32)
    with pytest.raises(ValueError, match="row 41 are not finite"):
        backend.logits(rows, None, np.array([3e38], F32), np.empty((2, 1), F32), 40)


@pytest.mark.parametrize("link", list(Link))
def test_outputs_are_the_reference(backend, link):
    rng = np.random.default_rng(1)
    k = 1 if link is Link.SIGMOID else 5
    z = (rng.standard_normal((40, k)) * 20).astype(F32)
    out = np.empty((40, 2 if k == 1 else k), F32)
    backend.outputs(z, "proba", link, out)
    np.testing.assert_array_equal(bits(out), bits(ref.proba(z, link)))
    labels = np.empty(40, np.int64)
    backend.outputs(z, "labels", None, labels)
    assert labels.tolist() == ref.labels(z).tolist()


def test_outputs_name_the_row_that_is_not_finite(backend):
    z = np.array([[0.0, 1.0], [np.nan, 1.0]], F32)
    with pytest.raises(ValueError, match="row 8 are not finite"):
        backend.outputs(z, "labels", None, np.empty(2, np.int64), 7)


Z = np.zeros((2, 3), F32)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((Z, None, np.zeros(3), np.zeros((2, 3), F32)), "bias must be a float32 vector"),
        ((Z, None, np.zeros(0, F32), np.zeros((2, 0), F32)), "bias must be a float32 vector"),
        ((Z[:, :2], None, np.zeros(3, F32), np.zeros((2, 3), F32)), r"rows must be .* \('n', 3\)"),
        ((Z, np.zeros((2, 3), F32), np.zeros(3, F32), np.zeros((2, 3), F32)), "weights must be"),
        ((Z, None, np.zeros(3, F32), np.zeros((3, 3), F32)), r"out must be a writable .* \(2, 3\)"),
        ((Z, None, np.zeros(3, F32), Z.copy()[:, ::-1]), "out must be"),
        ((Z, None, np.zeros(3, F32), Z.view()), "out may be the rows themselves"),
    ],
)
def test_logits_check_their_arguments(backend, args, match):
    with pytest.raises(ValueError, match=match):
        backend.logits(*args)


def test_logits_with_weights_write_apart_from_the_rows(backend):
    rows = np.zeros((2, 3), F32)
    with pytest.raises(ValueError, match="out may be the rows themselves"):
        backend.logits(rows, np.zeros((3, 3), F32), np.zeros(3, F32), rows)


def test_logits_check_first_row(backend):
    with pytest.raises(ValueError, match="first_row"):
        backend.logits(Z, None, np.zeros(3, F32), np.zeros((2, 3), F32), -1)
    bias = np.zeros(3, F32)
    with pytest.raises(ValueError, match="weights or the bias"):
        backend.logits(Z[:1].copy(), None, bias, bias.reshape(1, 3))


@pytest.mark.parametrize(
    ("kind", "link", "logits", "out", "match"),
    [
        ("proba", Link.SIGMOID, Z, np.zeros((2, 2), F32), "one logit"),
        ("proba", Link.OVR, Z[:, :1].copy(), np.zeros((2, 2), F32), "at least two"),
        ("proba", "softmax@1", Z, np.zeros((2, 3), F32), "link must be"),
        ("proba", Link.SOFTMAX, Z, np.zeros((2, 2), F32), "out must be"),
        ("labels", None, Z, np.zeros(2, np.int32), "int64 vector"),
        ("decision", None, Z, np.zeros(2, np.int64), "kind must be"),
        ("labels", None, Z.astype(np.float64), np.zeros(2, np.int64), "logits must be"),
        ("proba", Link.SOFTMAX, Z, Z, "share memory with the logits"),
    ],
)
def test_outputs_check_their_arguments(backend, kind, link, logits, out, match):
    with pytest.raises(ValueError, match=match):
        backend.outputs(logits, kind, link, out)
