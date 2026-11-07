"""Meta-tests of the int8 prefilter: fixed witnesses that fail for a weakened rule.

Each test runs a witness on the rule as it is (the control: ids and score bits of the exact
path, or the fallback the rule demands) and on the rule with one mistake (the weakened run must
return other ids or bits, or lack the fallback), so a comparison that stopped checking would not
pass unnoticed. docs/architecture/numerical-contract.md, "int8 prefilter (no new arithmetic)",
names the rules.
"""

import numpy as np
import pytest

from fse import _reference as exact_path
from fse import _reference_prefilter as prefilter
from support.prefilter import exact, prepared, same_result, search
from support.prefilter_witnesses import (
    TILT,
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


def lost(stored, inverse, query, k=1, **options):
    """Whether the search loses the exact top rows (other ids or score bits)."""
    ids, scores = exact(stored, inverse, query, k)
    return not same_result(search(stored, inverse, query, k, **options), ids, scores)


def test_the_witness_pair_has_the_geometry_the_tests_rely_on():
    stored, inverse, query, g = pair_store()
    _, qcodes, qscales, _, _ = prepared(query)
    a, _ = prefilter.estimates(stored, inverse, qcodes, qscales)
    c = exact_path.cosines(stored, inverse, query)[:, 0].astype(np.float64)
    gap = (a[9, 0] - a[5, 0]) / g.eps
    assert 1.85 < gap < 1.95  # below 2 eps, above 1.8 eps and above E = 1.0116 eps
    assert c[5] > c[9] > np.delete(c, [5, 9]).max()
    assert exact_path.top_k(c.astype(F32), 1).tolist() == [5]
    assert TILT == 2.34


def test_control_the_rule_keeps_the_top_row_of_the_pair():
    stored, inverse, query, _ = pair_store()
    assert not lost(stored, inverse, query)
    assert search(stored, inverse, query, 1).fallback is None


@pytest.mark.parametrize(
    ("slack", "absolute"),
    [(0.9, 0.0), (0.0, 0.0), (0.0, 2.0**-14)],
    ids=["0.9 eps", "zero", "tiny"],
)
def test_a_bound_below_the_error_of_the_query_loses_the_top_row(monkeypatch, slack, absolute):
    stored, inverse, query, _ = pair_store()
    monkeypatch.setattr(prefilter, "SLACK", slack)
    monkeypatch.setattr(prefilter, "ABSOLUTE", absolute)
    assert lost(stored, inverse, query)


def test_the_threshold_without_the_slack_loses_the_top_row(monkeypatch):
    stored, inverse, query, _ = pair_store()
    monkeypatch.setattr(prefilter, "floor_of", lambda threshold, _: threshold)
    assert lost(stored, inverse, query)


def test_a_floor_below_the_threshold_by_less_than_the_slack_loses_it_too(monkeypatch):
    stored, inverse, query, _ = pair_store()
    monkeypatch.setattr(prefilter, "floor_of", lambda threshold, bound: threshold - bound / 2)
    assert lost(stored, inverse, query)


def test_control_a_tie_at_the_threshold_stays(monkeypatch):
    tight_bounds(monkeypatch)
    stored, inverse, units, qcodes, qscales = tie_store()
    slack, ok = prefilter.bounds(units, qcodes, qscales)
    assert ok[0]
    assert slack[0] == 0.0625
    a, _ = prefilter.estimates(stored, inverse, qcodes, qscales)
    c = exact_path.cosines(stored, inverse, units)[:, 0]
    assert a[:, 0].tolist() == [-0.0625, 0.0625]
    assert (c == 0).all()
    assert a[0, 0] + slack[0] == a[1, 0] - slack[0]  # the tie
    got = prefilter.int8_pass(stored, inverse, qcodes, qscales, slack, np.full(1, -np.inf), 1)
    threshold = prefilter.kth_estimate(got.heap)
    kept = prefilter.survivors(got.rows[0], got.upper[0], float(threshold[0]), float(slack[0]))
    assert kept.tolist() == [0, 1]  # position 0 wins the tie of the scores


def survivors_of_exact_tie():
    stored, inverse, query = exact_tie_store()
    _, qcodes, qscales, slack, ok = prepared(query)
    assert ok[0]
    assert slack[0] == 2.0**-14  # eps_hat = 0
    a, _ = prefilter.estimates(stored, inverse, qcodes, qscales)
    assert a[:, 0].tolist() == [63 / 8192, 1 / 128]
    assert a[0, 0] + slack[0] == 127 / 16384  # the tie at the floor
    assert a[1, 0] - slack[0] == 127 / 16384
    got = prefilter.int8_pass(stored, inverse, qcodes, qscales, slack, np.full(1, -np.inf), 1)
    threshold = float(prefilter.kth_estimate(got.heap)[0])
    return prefilter.survivors(got.rows[0], got.upper[0], threshold, float(slack[0]))


def test_control_a_tie_at_the_floor_stays_under_the_fixed_constants():
    assert survivors_of_exact_tie().tolist() == [0, 1]


def test_an_exclusive_comparison_changes_the_survivors_at_a_tie_under_the_fixed_constants(
    monkeypatch,
):
    monkeypatch.setattr(prefilter, "keeps", lambda upper, floor: upper > floor)
    assert survivors_of_exact_tie().tolist() == [1]


def test_an_exclusive_comparison_loses_a_tie_at_the_threshold(monkeypatch):
    tight_bounds(monkeypatch)
    monkeypatch.setattr(prefilter, "keeps", lambda upper, floor: upper > floor)
    stored, inverse, units, qcodes, qscales = tie_store()
    slack, _ = prefilter.bounds(units, qcodes, qscales)
    got = prefilter.int8_pass(stored, inverse, qcodes, qscales, slack, np.full(1, -np.inf), 1)
    kept = prefilter.survivors(
        got.rows[0], got.upper[0], float(prefilter.kth_estimate(got.heap)[0]), float(slack[0])
    )
    assert kept.tolist() == [1]  # row 0 is the exact top row (a tie broken by position)
    scores = exact_path.cosines(stored[kept], inverse[kept], units)[:, 0]
    ids = kept[exact_path.top_k(scores, 1)]
    assert (
        ids.tolist()
        != exact_path.top_k(exact_path.cosines(stored, inverse, units)[:, 0], 1).tolist()
    )


def test_control_an_unsafe_row_sends_the_whole_call_to_the_exact_path():
    stored, inverse, query, row = unsafe_store()
    _, qcodes, qscales, _, _ = prepared(query)
    a, safe = prefilter.estimates(stored, inverse, qcodes, qscales)
    assert np.flatnonzero(~safe).tolist() == [row]
    assert a[row, 0] > 7
    assert exact_path.cosines(stored, inverse, query)[row, 0] < 0.5
    got = search(stored, inverse, query, 1)
    assert got.fallback == "unsafe_row"
    assert exact_path.top_k(exact_path.cosines(stored, inverse, query)[:, 0], 1).tolist() == [10]


def test_without_the_row_test_an_inflated_estimate_loses_the_top_row(monkeypatch):
    stored, inverse, query, _ = unsafe_store()
    monkeypatch.setattr(prefilter, "row_safe", lambda inv, _: np.isfinite(inv))
    got = search(stored, inverse, query, 1)
    assert got.fallback is None
    assert lost(stored, inverse, query)


def test_an_unsafe_row_in_the_heap_loses_the_top_row():
    stored, inverse, query, _ = unsafe_store()

    def unsafe_in_heap(stored, inverse, qcodes, qscales, slack, threshold, k):
        """The pass of the reference, except that an unsafe row enters the heap (not the list)."""
        estimate, safe = prefilter.estimates(stored, inverse, qcodes, qscales)
        heap = np.full((1, k), -np.inf)
        top, kept, tops = prefilter._scan(
            estimate[:, 0].tolist(), float(slack[0]), float(threshold[0]), k
        )
        heap[0, : len(top)] = top
        keep = [(r, t) for r, t in zip(kept, tops, strict=True) if safe[r]]
        rows = np.array([r for r, _ in keep], dtype=np.int64)
        return prefilter.Pass("ok", None, heap, (rows,), (np.array([t for _, t in keep]),))

    got = search(stored, inverse, query, 1, run_pass=unsafe_in_heap)
    assert got.fallback is None
    assert lost(stored, inverse, query, run_pass=unsafe_in_heap)


def test_control_a_query_with_underflowing_squares_takes_the_exact_path():
    stored, inverse = random_store(40, 5)
    query = small_norm_query()
    assert search(stored, inverse, query, 3).fallback == "query_range"


def test_without_the_query_test_the_call_is_not_sent_to_the_exact_path(monkeypatch):
    stored, inverse = random_store(40, 5)
    monkeypatch.setattr(prefilter, "query_ok", lambda norm2: np.isfinite(norm2))
    assert search(stored, inverse, small_norm_query(), 3).fallback is None


def test_control_the_survivors_do_not_depend_on_the_order_of_the_blocks():
    stored, inverse = random_store(512, 11)
    queries = np.random.default_rng(12).standard_normal((2, 256)).astype(F32)
    runs = [search(stored, inverse, queries, 3, block=64, order=order) for order in orders()]
    for other in runs[1:]:
        for a, b in zip(runs[0].rows, other.rows, strict=True):
            np.testing.assert_array_equal(a, b)


def orders():
    return [None, [7, 6, 5, 4, 3, 2, 1, 0], [3, 1, 7, 5, 0, 2, 4, 6]]


def test_a_cut_against_the_running_threshold_instead_of_the_final_one_depends_on_the_order():
    stored, inverse = random_store(512, 11)
    queries = np.random.default_rng(12).standard_normal((2, 256)).astype(F32)
    runs = [
        search(stored, inverse, queries, 3, block=64, order=order, compact="none")
        for order in orders()
    ]
    sizes = {tuple(len(rows) for rows in run.rows) for run in runs}
    assert len(sizes) > 1  # the sets differ, though every run has the same exact top rows
    ids, scores = exact(stored, inverse, queries, 3)
    assert all(same_result(run, ids, scores) for run in runs)


def test_a_weakened_bound_loses_nothing_on_random_rows(monkeypatch):
    """Control of the witnesses: only the fixed ones show a bound of 0.9 eps."""
    monkeypatch.setattr(prefilter, "SLACK", 0.9)
    monkeypatch.setattr(prefilter, "ABSOLUTE", 0.0)
    stored, inverse = random_store(300, 31)
    query = Geometry(31).query
    ids, scores = exact(stored, inverse, query, 5)
    assert same_result(search(stored, inverse, query, 5), ids, scores)
