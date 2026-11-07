"""The meta-tests of the int8 prefilter against the native variants.

The fixed witnesses of tests/test_reference_prefilter_meta.py (tests/support/prefilter_witnesses.py)
run here with the kernel's ``int8_bounds`` and ``int8_pass`` in place of the reference's: the
rule as it is must keep the exact top rows, send unsafe rows and queries outside the domain to
the exact path and keep the ties; and a slack below the error of the query, fed to the kernel,
must lose the top row of the witness, so the witnesses still bite on the native code. The
mutations of the other rules (an exclusive comparison, a threshold without the slack, no row
test) live in the reference and are weakened there; the kernel has one fixed rule.
"""

import numpy as np
import pytest

from fse import _native
from fse import _reference as exact_path
from fse import _reference_prefilter as prefilter
from support.prefilter import exact, prepared, same_result, search
from support.prefilter_native import pass_of, runnable_variants
from support.prefilter_witnesses import (
    Geometry,
    exact_tie_store,
    pair_store,
    random_store,
    small_norm_query,
    tie_store,
    tight_bounds,
    unsafe_store,
)

F32 = np.float32
VARIANTS = runnable_variants()


def native_bounds(units, qcodes, qscales):
    return _native.int8_bounds(units, qcodes, qscales)


def weakened(factor, absolute):
    """The kernel's slack with the error ``eps`` scaled by ``factor`` and ``absolute`` added."""

    def bounds(units, qcodes, qscales):
        slack, ok = native_bounds(units, qcodes, qscales)
        eps = (slack - prefilter.ABSOLUTE) / prefilter.SLACK
        return factor * eps + absolute, ok

    return bounds


def run(variant, stored, inverse, query, k=1, **options):
    return search(
        stored,
        inverse,
        query,
        k,
        run_pass=pass_of(variant),
        bounds=options.pop("bounds", native_bounds),
        **options,
    )


def lost(variant, stored, inverse, query, k=1, **options):
    ids, scores = exact(stored, inverse, query, k)
    return not same_result(run(variant, stored, inverse, query, k, **options), ids, scores)


@pytest.mark.parametrize("variant", VARIANTS)
def test_control_the_kernel_keeps_the_top_row_of_the_pair(variant):
    stored, inverse, query, _ = pair_store()
    assert not lost(variant, stored, inverse, query)
    assert run(variant, stored, inverse, query).fallback is None


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    ("factor", "absolute"),
    [(0.9, 0.0), (0.0, 0.0), (0.0, 2.0**-14)],
    ids=["0.9 eps", "zero", "tiny"],
)
def test_a_slack_below_the_error_of_the_query_loses_the_top_row(variant, factor, absolute):
    stored, inverse, query, _ = pair_store()
    assert lost(variant, stored, inverse, query, bounds=weakened(factor, absolute))


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_weakened_slack_loses_nothing_on_random_rows(variant):
    """Control of the witnesses: only the fixed pair shows a slack of 0.9 eps."""
    stored, inverse = random_store(300, 31)
    query = Geometry(31).query
    ids, scores = exact(stored, inverse, query, 5)
    got = run(variant, stored, inverse, query, 5, bounds=weakened(0.9, 0.0))
    assert same_result(got, ids, scores)


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_tie_at_the_threshold_stays_in_the_kernel(variant, monkeypatch):
    tight_bounds(monkeypatch)
    stored, inverse, units, qcodes, qscales = tie_store()
    slack, ok = prefilter.bounds(units, qcodes, qscales)  # E = eps_hat exactly
    assert ok[0]
    got = pass_of(variant)(stored, inverse, qcodes, qscales, slack, np.full(1, -np.inf), 1)
    final = float(prefilter.kth_estimate(got.heap)[0])
    kept = prefilter.survivors(got.rows[0], got.upper[0], final, float(slack[0]))
    assert kept.tolist() == [0, 1]  # position 0 wins the tie of the scores


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_tie_at_the_floor_stays_in_the_kernel_under_the_fixed_constants(variant):
    stored, inverse, query = exact_tie_store()
    _, qcodes, qscales, slack, ok = prepared(query)
    assert ok[0]
    assert native_bounds(*prepared(query)[:3])[0][0] == slack[0] == 2.0**-14
    got = pass_of(variant)(stored, inverse, qcodes, qscales, slack, np.full(1, -np.inf), 1)
    final = float(prefilter.kth_estimate(got.heap)[0])
    kept = prefilter.survivors(got.rows[0], got.upper[0], final, float(slack[0]))
    assert kept.tolist() == [0, 1]
    assert got.rows[0].tolist() == [0, 1]


@pytest.mark.parametrize("variant", VARIANTS)
def test_an_unsafe_row_sends_the_whole_call_to_the_exact_path(variant):
    stored, inverse, query, row = unsafe_store()
    got = run(variant, stored, inverse, query)
    assert got.fallback == "unsafe_row"
    ids, scores = exact(stored, inverse, query, 1)
    assert same_result(got, ids, scores)
    assert exact_path.top_k(exact_path.cosines(stored, inverse, query)[:, 0], 1).tolist() == [10]
    _, qcodes, qscales, slack, _ = prepared(query)
    args = (stored, inverse, qcodes, qscales, slack, np.full(1, -np.inf), 1)
    assert pass_of(variant)(*args).row == row


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_query_with_underflowing_squares_takes_the_exact_path(variant):
    stored, inverse = random_store(40, 5)
    assert run(variant, stored, inverse, small_norm_query(), 3).fallback == "query_range"


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_survivors_of_the_kernel_do_not_depend_on_the_blocks_or_their_order(variant):
    stored, inverse = random_store(512, 11)
    queries = np.random.default_rng(12).standard_normal((2, 256)).astype(F32)
    orders = [None, [7, 6, 5, 4, 3, 2, 1, 0], [3, 1, 7, 5, 0, 2, 4, 6]]
    runs = [run(variant, stored, inverse, queries, 3, block=64, order=o) for o in orders]
    runs.append(run(variant, stored, inverse, queries, 3, block=512))
    for other in runs[1:]:
        for a, b in zip(runs[0].rows, other.rows, strict=True):
            np.testing.assert_array_equal(a, b)
    reference = search(stored, inverse, queries, 3, block=64)
    for a, b in zip(runs[0].rows, reference.rows, strict=True):
        np.testing.assert_array_equal(a, b)
    assert all(len(rows) < 512 for rows in runs[0].rows)


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_cut_against_the_running_threshold_depends_on_the_order_in_the_kernel(variant):
    stored, inverse = random_store(512, 11)
    queries = np.random.default_rng(12).standard_normal((2, 256)).astype(F32)
    orders = [None, [7, 6, 5, 4, 3, 2, 1, 0], [3, 1, 7, 5, 0, 2, 4, 6]]
    runs = [
        run(variant, stored, inverse, queries, 3, block=64, order=o, compact="none") for o in orders
    ]
    assert len({tuple(len(rows) for rows in r.rows) for r in runs}) > 1
    ids, scores = exact(stored, inverse, queries, 3)
    assert all(same_result(r, ids, scores) for r in runs)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dim", [8, 256])
@pytest.mark.parametrize("k", [1, 3, 50])
def test_the_search_with_the_kernel_gives_the_bits_of_the_exact_path(variant, dim, k):
    stored, inverse = random_store(130, k + dim, dim=dim)
    queries = np.random.default_rng(k).standard_normal((3, dim)).astype(F32)
    ids, scores = exact(stored, inverse, queries, k)
    for block, order in [(130, None), (16, None), (16, [8, 7, 6, 5, 4, 3, 2, 1, 0]), (7, None)]:
        got = run(variant, stored, inverse, queries, k, block=block, order=order)
        assert got.fallback is None
        assert same_result(got, ids, scores)
