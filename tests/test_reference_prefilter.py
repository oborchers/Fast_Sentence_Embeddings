"""The reference of the int8 prefilter: bounds, estimates, the pass, the cut and the whole search."""

from decimal import Context, Decimal
from fractions import Fraction

import numpy as np
import pytest

from fse import _reference as exact_path
from fse import _reference_prefilter as prefilter
from support.prefilter import exact, int8_store, prepared, same_result, search
from support.prefilter_witnesses import (
    Geometry,
    int8_rows,
    pair_store,
    random_store,
    small_norm_query,
)

F32 = np.float32
DIGITS = Context(prec=60)


def exact_eps(units, qcodes, qscales):
    """``||u - s * qcodes||`` of one query from exact fractions, to 60 digits."""
    total = Fraction(0)
    for u, code in zip(units.tolist(), qcodes.tolist(), strict=True):
        total += (Fraction(u) - Fraction(float(qscales)) * code) ** 2
    return DIGITS.sqrt(DIGITS.divide(Decimal(total.numerator), Decimal(total.denominator)))


def one_hot(index, dim=256):
    query = np.zeros((1, dim), F32)
    query[0, index] = 1.0
    return query


@pytest.mark.parametrize("dim", [1, 16, 256, 1024])
def test_bounds_follow_their_definition(dim):
    queries = np.random.default_rng(dim).standard_normal((3, dim)).astype(F32)
    units, qcodes, qscales, slack, ok = prepared(queries)
    assert ok.all()
    for j in range(3):
        eps = exact_eps(units[j], qcodes[j], qscales[j])
        want = 1.003 * float(eps) + 2.0**-14
        assert abs(slack[j] - want) <= 2.0**-40 * want  # the float64 grouping of the proof
        # eps_hat of the contract of the proof: relative error below 2^-40
        eps_hat = (slack[j] - 2.0**-14) / 1.003
        assert abs(Decimal(eps_hat) - eps) / eps < Decimal(2) ** -40


def test_bounds_use_the_grouping_of_the_rule():
    units, qcodes, qscales, slack, _ = prepared(np.random.default_rng(5).standard_normal((1, 40)))
    total, norm = 0.0, 0.0
    for u, code in zip(units[0].tolist(), qcodes[0].tolist(), strict=True):
        total += (u - float(qscales[0]) * code) ** 2
        norm += u * u
    assert slack[0] == 1.003 * total**0.5 + 2.0**-14
    assert prefilter.bounds(units, qcodes, qscales)[1][0] == (norm <= 1.002)


def test_a_query_with_squares_that_underflow_is_outside_the_domain():
    units, _, _, _, ok = prepared(small_norm_query())
    assert float((units[0].astype(np.float64) ** 2).sum()) == 1.25
    assert not ok[0]


@pytest.mark.parametrize(("scale", "expected"), [(1.00099, True), (1.0011, False), (2.0, False)])
def test_the_norm_limit_of_the_query_is_1_002(scale, expected):
    units = one_hot(0) * F32(scale)
    codes, scales = exact_path.quantize_rows(units, "int8")
    assert prefilter.bounds(units, codes, scales)[1][0] == expected


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_query_that_is_not_finite_is_outside_the_domain(bad):
    units = one_hot(0)
    codes, scales = exact_path.quantize_rows(units, "int8")
    units[0, 3] = bad
    with np.errstate(invalid="ignore"):
        assert not prefilter.bounds(units, codes, scales)[1][0]


def test_bounds_refuse_what_is_outside_their_proof():
    units, qcodes, qscales, _, _ = prepared(np.ones((2, 8), F32))
    bounds = prefilter.bounds
    with pytest.raises(ValueError, match="at most 1024"):
        bounds(np.ones((1, 1025), F32), np.zeros((1, 1025), np.int8), np.ones(1, F32))
    with pytest.raises(ValueError, match=r"\[-127, 127\]"):
        bounds(units, np.full_like(qcodes, -128), qscales)
    with pytest.raises(ValueError, match="int8 matrix"):
        bounds(units, qcodes.astype(np.int16), qscales)
    with pytest.raises(ValueError, match="float32 scale per query"):
        bounds(units, qcodes, qscales.astype(np.float64))
    with pytest.raises(ValueError, match="finite and not negative"):
        bounds(units, qcodes, np.array([1.0, -1.0], F32))
    with pytest.raises(ValueError, match="finite and not negative"):
        bounds(units, qcodes, np.array([1.0, np.inf], F32))
    with pytest.raises(ValueError, match="float32 queries"):
        bounds(units.astype(np.float64), qcodes, qscales)


def test_estimates_are_the_integer_sums_with_the_grouping_of_the_rule():
    stored, inverse = random_store(40, 3, dim=64)
    _, qcodes, qscales, _, _ = prepared(np.random.default_rng(4).standard_normal((3, 64)))
    a, safe = prefilter.estimates(stored, inverse, qcodes, qscales)
    assert safe.all()
    for r in range(len(stored)):
        for j in range(3):
            total = sum(int(x) * int(y) for x, y in zip(stored[r], qcodes[j], strict=True))
            assert a[r, j] == (float(qscales[j]) * total) * float(inverse[r])


def test_the_largest_sums_are_exact_integers():
    stored = np.full((2, 1024), -128, np.int8)
    stored[1] = 127
    codes = np.full((1, 1024), -127, np.int8)
    inverse = exact_path.inverse_norms(stored)
    a, safe = prefilter.estimates(stored, inverse, codes, np.ones(1, F32))
    assert safe.all()
    assert a[0, 0] == (1.0 * (128 * 127 * 1024)) * float(inverse[0])
    assert a[1, 0] == (1.0 * (-127 * 127 * 1024)) * float(inverse[1])
    assert 128 * 127 * 1024 < 2**24
    assert 128 * 128 * 1024 == 2**24


def estimates_of(inverse_values, stored, dim=2):
    codes = np.ones((1, dim), np.int8)
    return prefilter.estimates(stored, np.asarray(inverse_values, F32), codes, np.full(1, 0.5, F32))


@pytest.mark.parametrize(
    ("inverse", "safe"),
    [
        (float(F32(1 / np.sqrt(2))), True),
        (1e-3, True),  # too small an inverse only shrinks every error
        (0.0, True),
        (float(F32(2 / np.sqrt(2))), False),  # a norm too large by a factor 2
        (float(F32(1.001 / np.sqrt(2))), False),
        (float(F32(1.0000006 / np.sqrt(2))), False),  # just outside
        (float(F32(1.0000002 / np.sqrt(2))), True),  # inside 1 + 2^-20
        (-0.5, False),
        (np.nan, False),
        (np.inf, False),
        (-np.inf, False),
    ],
)
def test_a_row_is_safe_only_with_a_norm_that_is_not_too_large(inverse, safe):
    stored = np.array([[1, 1]], np.int8)  # S = 2
    with np.errstate(invalid="ignore"):
        assert estimates_of([inverse], stored)[1][0] == safe


def test_the_limit_of_the_row_norm_is_1_plus_2_to_the_minus_20():
    inv = np.float64(1.0)  # S = 1
    below = prefilter.row_safe(inv, np.float64(1.0 + 2.0**-20))
    above = prefilter.row_safe(inv, np.float64(1.0 + 2.0**-19))
    assert bool(below)
    assert not bool(above)


def test_zero_rows_are_safe_whatever_their_finite_inverse():
    stored = np.zeros((3, 4), np.int8)
    codes = np.ones((1, 4), np.int8)
    inverse = np.array([0.0, 1e30, np.inf], F32)
    a, safe = prefilter.estimates(stored, inverse, codes, np.ones(1, F32))
    assert safe.tolist() == [True, True, False]
    assert (a[:2] == 0).all()


def test_estimates_refuse_shapes_and_types_they_do_not_prove():
    stored, inverse = random_store(4, 1, dim=8)
    codes = np.ones((1, 8), np.int8)
    scale = np.ones(1, F32)
    with pytest.raises(ValueError, match="int8 matrix with the dimension"):
        prefilter.estimates(stored.astype(np.int16), inverse, codes, scale)
    with pytest.raises(ValueError, match="int8 matrix with the dimension"):
        prefilter.estimates(stored, inverse, np.ones((1, 9), np.int8), scale)
    with pytest.raises(ValueError, match="one float32 inverse norm"):
        prefilter.estimates(stored, inverse[:3], codes, scale)


def run_pass(stored, inverse, g, k, threshold=-np.inf):
    return prefilter.int8_pass(
        stored,
        inverse,
        g.qcodes,
        g.qscales,
        g.slack,
        np.full(len(g.qcodes), threshold),
        k,
    )


def test_the_pass_keeps_the_k_largest_estimates_in_a_sorted_heap():
    stored, inverse = random_store(50, 2)
    g = Geometry(2)
    got = run_pass(stored, inverse, g, 4)
    a, _ = prefilter.estimates(stored, inverse, g.qcodes, g.qscales)
    assert got.status == "ok"
    np.testing.assert_array_equal(got.heap[0], np.sort(a[:, 0])[::-1][:4])
    short = run_pass(stored[:3], inverse[:3], g, 4)
    assert short.heap[0, 3] == -np.inf
    assert np.isfinite(short.heap[0, :3]).all()
    assert prefilter.kth_estimate(short.heap)[0] == -np.inf


def test_the_first_k_rows_are_kept_and_later_rows_need_the_running_threshold():
    stored, inverse = random_store(300, 6)
    g = Geometry(6)
    got = run_pass(stored, inverse, g, 5)
    a, _ = prefilter.estimates(stored, inverse, g.qcodes, g.qscales)
    assert got.rows[0][:5].tolist() == [0, 1, 2, 3, 4]
    assert (np.diff(got.rows[0]) > 0).all()
    np.testing.assert_array_equal(got.upper[0], a[got.rows[0], 0] + g.slack[0])
    final = prefilter.survivors(
        got.rows[0], got.upper[0], float(prefilter.kth_estimate(got.heap)[0]), float(g.slack[0])
    )
    assert set(final.tolist()) <= set(got.rows[0].tolist())
    assert len(got.rows[0]) < 300


def test_an_incoming_threshold_can_only_remove_rows():
    stored, inverse = random_store(300, 8)
    g = Geometry(8)
    free = run_pass(stored, inverse, g, 3)
    bound = float(prefilter.kth_estimate(free.heap)[0])  # the k-th largest of all rows
    tight = run_pass(stored, inverse, g, 3, threshold=bound)
    assert set(tight.rows[0].tolist()) <= set(free.rows[0].tolist())
    assert len(tight.rows[0]) < len(free.rows[0])
    final = prefilter.survivors(free.rows[0], free.upper[0], bound, float(g.slack[0]))
    np.testing.assert_array_equal(
        final, prefilter.survivors(tight.rows[0], tight.upper[0], bound, float(g.slack[0]))
    )


def test_the_pass_reports_the_first_unsafe_row_and_nothing_else():
    stored, inverse = random_store(20, 1)
    g = Geometry(1)
    inverse = inverse.copy()
    inverse[[7, 12]] = [np.nan, 2 * inverse[12]]
    got = run_pass(stored, inverse, g, 3)
    assert (got.status, got.row) == ("unsafe", 7)
    assert got.heap is None
    assert got.rows is None
    assert got.upper is None


def test_the_pass_checks_its_arguments():
    stored, inverse = random_store(5, 1)
    g = Geometry(1)

    def call(slack=g.slack, threshold=None, k=2):
        threshold = np.full(1, -np.inf) if threshold is None else threshold
        return prefilter.int8_pass(stored, inverse, g.qcodes, g.qscales, slack, threshold, k)

    assert call().status == "ok"
    with pytest.raises(ValueError, match="positive integer"):
        call(k=0)
    with pytest.raises(ValueError, match="positive integer"):
        call(k=True)
    with pytest.raises(ValueError, match="slack must be one float64"):
        call(slack=g.slack.astype(F32))
    with pytest.raises(ValueError, match="finite and not negative"):
        call(slack=np.array([-1.0]))
    with pytest.raises(ValueError, match="threshold must be one float64"):
        call(threshold=np.zeros(2))
    with pytest.raises(ValueError, match="must not be NaN"):
        call(threshold=np.array([np.nan]))


def test_heaps_merge_to_the_k_largest_of_all():
    first = np.array([[5.0, 3.0, -np.inf]])
    second = np.array([[4.0, 4.0, 1.0]])
    merged = prefilter.merge_heaps([first, second], 3)
    np.testing.assert_array_equal(merged, [[5.0, 4.0, 4.0]])
    assert prefilter.kth_estimate(merged)[0] == 4.0
    assert prefilter.merge_heaps([first], 3).tolist() == first.tolist()


@pytest.mark.parametrize("dim", [8, 256])
@pytest.mark.parametrize("k", [1, 3, 50])
def test_the_search_gives_the_bits_of_the_exact_path(dim, k):
    stored, inverse = random_store(130, k + dim, dim=dim)
    queries = np.random.default_rng(k).standard_normal((3, dim)).astype(F32)
    ids, scores = exact(stored, inverse, queries, k)
    for block, order in [(130, None), (16, None), (16, [8, 7, 6, 5, 4, 3, 2, 1, 0]), (7, None)]:
        got = search(stored, inverse, queries, k, block=block, order=order)
        assert got.fallback is None
        assert same_result(got, ids, scores)
        assert all(len(i) == min(k, 130) for i in got.ids)


def test_the_survivors_do_not_depend_on_the_blocks_or_their_order():
    stored, inverse = random_store(512, 11)
    queries = np.random.default_rng(12).standard_normal((2, 256)).astype(F32)
    runs = [
        search(stored, inverse, queries, 3, block=block, order=order)
        for block, order in [(512, None), (64, None), (64, [7, 6, 5, 4, 3, 2, 1, 0]), (50, None)]
    ]
    for other in runs[1:]:
        for a, b in zip(runs[0].rows, other.rows, strict=True):
            np.testing.assert_array_equal(a, b)
    assert all(len(rows) < 512 for rows in runs[0].rows)


def test_equal_rows_and_zero_rows_and_ties_at_the_kth_value():
    rng = np.random.default_rng(13)
    base = int8_rows(rng.standard_normal((6, 64)))
    stored, inverse = int8_store(np.concatenate([base, base, np.zeros((4, 64), np.int8), base]))
    queries = np.concatenate([base[:2].astype(F32), -base[3:4].astype(F32)])
    for k in (1, 6, 7, 40):
        ids, scores = exact(stored, inverse, queries, k)
        got = search(stored, inverse, queries, k, block=5)
        assert got.fallback is None
        assert same_result(got, ids, scores)
    got = search(stored, inverse, queries, 1)
    assert got.ids[0].tolist() == [0]  # the smaller position of the equal rows wins


def test_the_extremes_of_the_integer_range_stay_inside_the_bound():
    dim = 1024
    pattern = np.where(np.arange(dim) % 2, 127, -128).astype(np.int8)
    rows = [np.full(dim, -128), np.full(dim, 127), pattern, -pattern - 1, np.zeros(dim)]
    hot = np.zeros(dim, np.int8)
    hot[0] = -128
    stored, inverse = int8_store(np.stack([*rows, hot]).astype(np.int8))
    queries = np.stack([np.ones(dim), np.where(np.arange(dim) % 2, 1.0, -1.0), -pattern]).astype(
        F32
    )
    _, qcodes, qscales, slack, ok = prepared(queries)
    assert ok.all()
    a, safe = prefilter.estimates(stored, inverse, qcodes, qscales)
    cosines = exact_path.cosines(stored, inverse, queries)
    assert safe.all()
    assert (np.abs(cosines.astype(np.float64) - a) <= slack).all()


def test_the_bound_holds_for_every_pair_and_is_nearly_reached(record_property):
    """V1: ``fl(a - E) <= c <= fl(a + E)``. The adversarial pair reaches 0.99 of ``eps_hat``."""
    worst = 0.0
    for dim, rows, seed in [(256, 96, 1), (1024, 40, 2), (64, 64, 3), (256, 40, 4)]:
        stored, inverse = random_store(rows, seed, dim=dim)
        queries = np.random.default_rng(seed).standard_normal((4, dim)).astype(F32)
        _, qcodes, qscales, slack, ok = prepared(queries)
        a, safe = prefilter.estimates(stored, inverse, qcodes, qscales)
        c = exact_path.cosines(stored, inverse, queries).astype(np.float64)
        assert ok.all()
        assert safe.all()
        assert (c <= a + slack).all()
        assert (c >= a - slack).all()
        worst = max(worst, float((np.abs(c - a) / slack).max()))
    stored, inverse, query, _ = pair_store()
    _, qcodes, qscales, slack, _ = prepared(query)
    a, _ = prefilter.estimates(stored, inverse, qcodes, qscales)
    c = exact_path.cosines(stored, inverse, query).astype(np.float64)
    reach = float((np.abs(c - a)[:, 0] / (slack[0] - 2.0**-14) * 1.003).max())
    assert (c <= a + slack).all()
    assert (c >= a - slack).all()
    assert reach > 0.95
    record_property("random_worst_gap_over_slack", worst)
    record_property("adversarial_gap_over_eps", reach)


def test_the_survivors_of_the_search_are_a_small_part_of_a_large_store(record_property):
    stored, inverse = random_store(2000, 21)
    queries = np.random.default_rng(22).standard_normal((2, 256)).astype(F32)
    got = search(stored, inverse, queries, 10, block=500)
    ids, scores = exact(stored, inverse, queries, 10)
    assert same_result(got, ids, scores)
    share = max(len(rows) for rows in got.rows) / 2000
    record_property("survivor_share", share)
    assert share < 0.5
