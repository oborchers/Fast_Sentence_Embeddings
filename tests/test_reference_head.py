"""The numpy reference of linear heads: exp@1, the links, labels, folding and logits."""

from fractions import Fraction

import numpy as np
import pytest

from fse import _reference_head as ref
from fse._contract import Link
from fse._reference import dot16, quantize_rows, score

F32 = np.float32
TINY = F32(np.finfo(F32).smallest_subnormal)


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def ulps(got, want):
    """|got - want| in float32 ulps of ``want`` (float64), the smallest subnormal for 0."""
    want = np.asarray(want, dtype=np.float64)
    step = np.spacing(np.abs(want.astype(F32))).astype(np.float64)
    step = np.where(step == 0, float(TINY), step)
    return np.abs(np.asarray(got, dtype=np.float64) - want) / step


def round_f32(value: Fraction) -> F32:
    """The float32 nearest to an exact rational, ties to even (finite, nonzero values)."""
    guess = F32(float(value))
    candidates = {guess, np.nextafter(guess, F32(np.inf)), np.nextafter(guess, F32(-np.inf))}
    return min(
        candidates,
        key=lambda c: (abs(Fraction(float(c)) - value), int(bits(c)) & 1),
    )


def exp_by_fractions(x: float) -> F32:
    """exp@1 of one value from its definition, every fused operation rounded exactly."""
    x = F32(x)
    if not x >= ref.EXP_LOW:
        return F32(0)

    def fma(a, b, c):
        exact = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
        return F32(0) if exact == 0 else round_f32(exact)

    k = F32(np.rint(F32(x * ref.LOG2E)))
    r = fma(-k, ref.LN2_HI, x)
    r = fma(-k, ref.LN2_LO, r)
    p = ref.EXP_POLY[-1]
    for c in ref.EXP_POLY[-2::-1]:
        p = fma(p, r, c)
    exact = Fraction(float(p)) * Fraction(2) ** int(k)
    smallest = Fraction(float(TINY))
    if exact < smallest * 2**23:  # subnormal: round to a multiple of the smallest one
        steps = exact / smallest
        whole = int(steps)
        frac = steps - whole
        whole += frac > Fraction(1, 2) or (frac == Fraction(1, 2) and whole % 2 == 1)
        return F32(whole * float(smallest))
    return round_f32(exact)


def test_exp_follows_its_definition_operation_by_operation():
    rng = np.random.default_rng(1)
    x = np.concatenate(
        [
            -rng.uniform(0, 104, 150),
            -rng.uniform(0, 1e-3, 30),
            [-103.9, -103.3, -103.0, -87.4, -87.3, -0.5 * np.log(2), -1e-38],
        ]
    ).astype(F32)
    got = ref.exp(x)
    want = np.array([exp_by_fractions(v) for v in x], dtype=F32)
    np.testing.assert_array_equal(bits(got), bits(want))


def test_exp_is_within_two_ulps_of_float64():
    rng = np.random.default_rng(2)
    x = np.concatenate(
        [-rng.uniform(0, 104, 200_000), -rng.exponential(1.0, 100_000), -rng.uniform(0, 1e-6, 1000)]
    ).astype(F32)
    assert ulps(ref.exp(x), np.exp(x.astype(np.float64))).max() <= 2


def test_exp_edges():
    x = np.array([0.0, -0.0, -104.0, np.nextafter(F32(-104), F32(-200)), -np.inf, -1e30], F32)
    got = ref.exp(x)
    assert got.tolist()[:2] == [1.0, 1.0]
    assert bits(got[3:]).tolist() == [0, 0, 0]  # +0.0 below -104
    assert got[2] == 0  # exp(-104) is below half the smallest subnormal
    sub = ref.exp(np.array([-103.0, -100.0, -88.0], F32))
    assert (sub > 0).all()
    assert (sub < np.finfo(F32).tiny).all()
    assert ulps(sub, np.exp(np.array([-103.0, -100.0, -88.0]))).max() <= 1


def test_sigmoid_of_both_signs_and_the_extremes():
    z = np.array([0.0, -0.0, 1.0, -1.0, 20.0, -20.0, 200.0, -200.0, 3e38, -3e38], F32)
    p = ref.sigmoid(z)
    assert p[0] == p[1] == F32(0.5)
    assert p[6] == p[8] == 1
    assert bits(p[[7, 9]]).tolist() == [0, 0]
    want = 1 / (1 + np.exp(-z[:6].astype(np.float64)))
    assert ulps(p[:6], want).max() <= 3
    e = ref.exp(np.array([-1.0], F32))[0]
    assert bits(p[3]) == bits(e / (F32(1) + e))  # z < 0 takes e / (1 + e)
    assert bits(p[2]) == bits(F32(1) / (F32(1) + e))


def test_softmax_rows():
    z = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [3e38, -3e38, 0.0], [-0.0, 0.0, -1.0]], F32)
    p = ref.softmax(z)
    assert p[1].tolist() == [F32(1) / F32(3)] * 3
    assert p[2].tolist() == [1.0, 0.0, 0.0]  # no overflow: z - max
    e = ref.exp(np.array([1.0 - 3.0, 2.0 - 3.0, 0.0], F32))
    s = F32(F32(F32(0) + e[0]) + e[1]) + e[2]  # the sum in index order
    np.testing.assert_array_equal(bits(p[0]), bits(e / s))
    assert p[3, 0] == p[3, 1]


def test_ovr_rows_and_the_uniform_row():
    z = np.array([[1.0, -2.0, 0.5], [-200.0, -300.0, -1e30]], F32)
    p = ref.ovr(z)
    s = ref.sigmoid(z[0])
    total = F32(F32(F32(0) + s[0]) + s[1]) + s[2]
    np.testing.assert_array_equal(bits(p[0]), bits(s / total))
    assert p[1].tolist() == [F32(1) / F32(3)] * 3


def test_proba_by_link():
    z1 = np.array([[0.3], [-2.0]], F32)
    p = ref.proba(z1, Link.SIGMOID)
    s = ref.sigmoid(z1[:, 0])
    np.testing.assert_array_equal(bits(p), bits(np.stack([F32(1) - s, s], axis=1)))
    z3 = np.array([[0.3, 1.0, -1.0]], F32)
    np.testing.assert_array_equal(bits(ref.proba(z3, Link.SOFTMAX)), bits(ref.softmax(z3)))
    np.testing.assert_array_equal(bits(ref.proba(z3, Link.OVR)), bits(ref.ovr(z3)))
    with pytest.raises(ValueError, match="one logit"):
        ref.proba(z3, Link.SIGMOID)
    with pytest.raises(ValueError, match="at least two"):
        ref.proba(z1, Link.OVR)
    pair = np.concatenate([-z1, z1], axis=1)  # one logit with softmax: softmax of [-z, z]
    np.testing.assert_array_equal(bits(ref.proba(z1, Link.SOFTMAX)), bits(ref.softmax(pair)))


def test_labels():
    z1 = np.array([[0.0], [-0.0], [TINY], [-1.0], [2.0]], F32)
    assert ref.labels(z1).tolist() == [0, 0, 1, 0, 1]
    z3 = np.array([[1.0, 3.0, 3.0], [-0.0, 0.0, -1.0], [-1.0, -2.0, -0.5]], F32)
    assert ref.labels(z3).tolist() == [1, 0, 2]  # ties to the smaller index
    assert ref.labels(z3).dtype == np.int64


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_fold_scores_every_row_with_every_head_row(dtype):
    rng = np.random.default_rng(3)
    base = rng.standard_normal((7, 20)).astype(F32)
    base[0] = -0.0
    stored, scales = quantize_rows(base, dtype)
    weight = rng.standard_normal((3, 20)).astype(F32)
    folded = ref.fold(stored, scales, weight)
    assert folded.shape == (7, 3)
    assert folded.dtype == F32
    for j in range(3):
        np.testing.assert_array_equal(bits(folded[:, j]), bits(score(stored, scales, weight[j])))
    assert bits(folded[0]).tolist() == [0, 0, 0]  # +0.0: the fold adds the bias 0


def test_logits_with_and_without_the_weights():
    rng = np.random.default_rng(4)
    rows = rng.standard_normal((5, 33)).astype(F32)
    weight = rng.standard_normal((2, 33)).astype(F32)
    bias = np.array([0.5, -0.0], F32)
    got = ref.logits(rows, weight, bias)
    for j in range(2):
        np.testing.assert_array_equal(bits(got[:, j]), bits(score(rows, None, weight[j], bias[j])))
        np.testing.assert_array_equal(bits(got[:, j]), bits(dot16(rows, weight[j]) + bias[j]))
    folded = rng.standard_normal((5, 2)).astype(F32)
    np.testing.assert_array_equal(bits(ref.logits(folded, None, bias)), bits(folded + bias))


def test_logits_that_are_not_finite_name_their_row():
    rows = np.array([[1.0], [3e38]], F32)
    with pytest.raises(ValueError, match="row 11 are not finite"):
        ref.logits(rows, None, np.array([3e38], F32), first_row=10)
    with pytest.raises(ValueError, match="row 1 are not finite"):
        ref.logits(rows, np.array([[4.0]], F32), np.array([0.0], F32))
