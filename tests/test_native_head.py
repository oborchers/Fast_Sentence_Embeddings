"""The head routines of the kernel (_native_head.pxi) against fse._reference_head, bit for bit."""

import numpy as np
import pytest

from fse import _native
from fse import _reference_head as ref
from fse._contract import Link

F32 = np.float32
TINY = F32(np.finfo(F32).smallest_subnormal)
ISAS = _native.available_isas()
ENVIRONMENTS = {"flush to zero": {"flush": True}, "round up": {"round_up": True}}


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _exp_inputs():
    rng = np.random.default_rng(0)
    return np.concatenate(
        [
            -rng.uniform(0, 110, 400_000),
            -rng.exponential(1.0, 100_000),
            -rng.uniform(0, 1e-6, 1000),
            [0.0, -0.0, -104.0, -103.97, -103.28, -87.33, -1e-45, -np.inf, -3e38],
        ]
    ).astype(F32)


def test_exp1_matches_the_reference():
    x = _exp_inputs()
    np.testing.assert_array_equal(bits(_native.exp1(x)), bits(ref.exp(x)))
    grid = x[:600].reshape(20, 30)
    assert _native.exp1(grid).shape == (20, 30)
    assert _native.exp1(np.zeros(0, F32)).shape == (0,)
    with pytest.raises(ValueError, match="float32 array"):
        _native.exp1(np.zeros(3))


def _logit_cases():
    rng = np.random.default_rng(1)
    rows = rng.standard_normal((40, 37)).astype(F32)
    rows[3] = -0.0
    rows[4] = TINY
    weights = rng.standard_normal((5, 37)).astype(F32)
    bias = np.array([0.5, -0.0, TINY, -2.0, 0.0], F32)
    folded = rng.standard_normal((40, 5)).astype(F32)
    folded[0] = -0.0
    folded[1] = -TINY
    return rows, weights, bias, folded


@pytest.mark.parametrize("isa", ISAS)
def test_head_logits_match_the_reference(isa):
    rows, weights, bias, folded = _logit_cases()
    out = np.empty((40, 5), F32)
    _native.head_logits(rows, weights, bias, out, isa=isa)
    np.testing.assert_array_equal(bits(out), bits(ref.logits(rows, weights, bias)))
    _native.head_logits(folded, None, bias, out, isa=isa)
    np.testing.assert_array_equal(bits(out), bits(ref.logits(folded, None, bias)))
    inplace = folded.copy()
    _native.head_logits(inplace, None, bias, inplace, isa=isa)  # out may be rows
    np.testing.assert_array_equal(bits(inplace), bits(out))


@pytest.mark.parametrize("weights", [None, np.array([[2.0]], F32)])
def test_head_logits_that_are_not_finite_name_their_row(weights):
    rows = np.array([[1.0], [2.0], [3e38], [3e38]], F32)
    bias = np.array([3e38], F32) if weights is None else np.array([0.0], F32)
    with pytest.raises(ValueError, match="row 7 are not finite"):
        _native.head_logits(rows, weights, bias, np.empty((4, 1), F32), first_row=5)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((np.zeros((2, 3), F32), None, np.zeros(2, F32), np.zeros((2, 2), F32)), "rows must be"),
        (
            (np.zeros((2, 3), F32), np.zeros((2, 4), F32), np.zeros(2, F32), np.zeros((2, 2), F32)),
            "weights must be",
        ),
        ((np.zeros((2, 2), F32), None, np.zeros(2), np.zeros((2, 2), F32)), "bias must be"),
        ((np.zeros((2, 2), F32), None, np.zeros(0, F32), np.zeros((2, 0), F32)), "bias must be"),
        (
            (np.zeros((2, 2), F32), None, np.zeros(2, F32), np.zeros((2, 2), F32)[:, :1]),
            "out must be",
        ),
        ((np.zeros(2, F32), None, np.zeros(2, F32), np.zeros((2, 2), F32)), "float32 matrix"),
    ],
)
def test_head_logits_check_their_arguments(args, match):
    with pytest.raises(ValueError, match=match):
        _native.head_logits(*args)


def _proba_cases():
    rng = np.random.default_rng(2)
    one = np.concatenate(
        [rng.standard_normal((50, 1)) * 30, [[0.0], [-0.0], [200.0], [-200.0], [3e38], [-3e38]]]
    ).astype(F32)
    many = (rng.standard_normal((60, 17)) * 40).astype(F32)
    many[0] = 5.0  # a tie of every logit
    many[1, :] = -200.0  # an ovr row whose sum is 0
    many[2, :2] = [3e38, -3e38]
    many[3, :3] = [-0.0, 0.0, -0.0]
    return one, many


def test_head_outputs_match_the_reference():
    one, many = _proba_cases()
    out = np.empty((len(one), 2), F32)
    _native.head_outputs(one, "proba", Link.SIGMOID, out)
    np.testing.assert_array_equal(bits(out), bits(ref.proba(one, Link.SIGMOID)))
    for link in (Link.SOFTMAX, Link.OVR):
        out = np.empty_like(many)
        _native.head_outputs(many, "proba", link, out)
        np.testing.assert_array_equal(bits(out), bits(ref.proba(many, link)), err_msg=str(link))
    for z in (one, many, many[:, :2]):
        labels = np.empty(len(z), np.int64)
        _native.head_outputs(np.ascontiguousarray(z), "labels", None, labels)
        assert labels.tolist() == ref.labels(z).tolist()


@pytest.mark.parametrize("kind", ["proba", "labels"])
def test_head_outputs_of_logits_that_are_not_finite_name_their_row(kind):
    z = np.array([[1.0, 2.0], [np.inf, 0.0], [np.nan, 1.0]], F32)
    out = np.empty((3, 2), F32) if kind == "proba" else np.empty(3, np.int64)
    with pytest.raises(ValueError, match="row 4 are not finite"):
        _native.head_outputs(z, kind, Link.SOFTMAX, out, first_row=3)


@pytest.mark.parametrize(
    ("kind", "link", "logits", "out", "match"),
    [
        ("proba", Link.SIGMOID, np.zeros((2, 2), F32), np.zeros((2, 2), F32), "one logit"),
        ("proba", Link.OVR, np.zeros((2, 1), F32), np.zeros((2, 2), F32), "at least two"),
        ("proba", "softmax@1", np.zeros((2, 2), F32), np.zeros((2, 2), F32), "link must be"),
        ("proba", Link.SOFTMAX, np.zeros((2, 2), F32), np.zeros((2, 3), F32), "out must be"),
        ("labels", None, np.zeros((2, 2), F32), np.zeros(2, np.int32), "int64 vector"),
        ("decision", None, np.zeros((2, 2), F32), np.zeros(2, np.int64), "kind must be"),
        ("labels", None, np.zeros((2, 2)), np.zeros(2, np.int64), "logits must be"),
        ("labels", None, np.zeros(2, F32), np.zeros(2, np.int64), "float32 matrix"),
    ],
)
def test_head_outputs_check_their_arguments(kind, link, logits, out, match):
    with pytest.raises(ValueError, match=match):
        _native.head_outputs(logits, kind, link, out)


def _environment_calls():
    """Every head entry point on inputs whose results depend on the environment."""
    x = np.array([-100.0, -103.5, -88.0, -0.3, -1e-30], F32)
    rows = np.array([[TINY * 8, 1.0, 1.0], [TINY * 4, 1.0, -1.0]], F32)
    bias = np.array([TINY * 3, 2.0**-30, -(2.0**-30)], F32)
    weights = np.full((3, 3), F32(0.5), F32)
    soft = np.array([[0.0, -95.0, -100.0], [1.0, 1.0 + 2**-20, -0.1]], F32)
    one = np.array([[-90.0], [0.1], [95.0]], F32)

    def logits(w, r):
        out = np.empty((2, 3), F32)
        _native.head_logits(r, w, bias, out)
        return out

    def outputs(z, link):
        out = np.empty((len(z), 2 if link is Link.SIGMOID else z.shape[1]), F32)
        _native.head_outputs(z, "proba", link, out)
        return out

    return {
        "exp1": (lambda: _native.exp1(x), lambda: ref.exp(x)),
        "bias": (lambda: logits(None, rows), lambda: ref.logits(rows, None, bias)),
        "scores": (lambda: logits(weights, rows), lambda: ref.logits(rows, weights, bias)),
        "softmax": (lambda: outputs(soft, Link.SOFTMAX), lambda: ref.proba(soft, Link.SOFTMAX)),
        "ovr": (lambda: outputs(soft, Link.OVR), lambda: ref.proba(soft, Link.OVR)),
        "sigmoid": (lambda: outputs(one, Link.SIGMOID), lambda: ref.proba(one, Link.SIGMOID)),
    }


def test_the_environment_cases_depend_on_the_environment():
    """The reference, which runs in the environment of its thread, changes on these
    inputs under one of the switches, so the next test can tell a kernel that inherits it."""
    calls = _environment_calls()
    want = {name: bits(reference()).tolist() for name, (_, reference) in calls.items()}
    changed = set()
    for env in ENVIRONMENTS.values():
        saved = _native._test_fpenv_set(**env)
        try:
            inside = {name: bits(reference()).tolist() for name, (_, reference) in calls.items()}
        finally:
            _native._test_fpenv_restore(saved)
        changed |= {name for name in calls if inside[name] != want[name]}
    assert changed == set(calls)


@pytest.mark.parametrize("name", list(_environment_calls()))
@pytest.mark.parametrize("env", ENVIRONMENTS, ids=list(ENVIRONMENTS))
def test_head_routines_ignore_and_restore_the_environment(env, name):
    call, reference = _environment_calls()[name]
    want = reference()
    saved = _native._test_fpenv_set(**ENVIRONMENTS[env])
    try:
        changed = _native._test_fpenv_get()
        got = call()
        after = _native._test_fpenv_get()
    finally:
        _native._test_fpenv_restore(saved)
    assert after == changed
    np.testing.assert_array_equal(bits(got), bits(want))
