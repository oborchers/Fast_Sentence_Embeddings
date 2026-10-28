"""The search by blocks: the best rows of a block, merged in any order, as top_k."""

import numpy as np
import pytest

from fse import _reference
from fse._store import _search
from fse._store._layout import Shard

F32 = np.float32


def _search_blocks(scores, k, order, size=16):
    """Merge the blocks of ``scores`` (rows, queries) in ``order``, each with the floors of
    the blocks merged before it."""
    candidates = _search.Candidates(scores.shape[1], k)
    for b in order:
        block = scores[b * size : (b + 1) * size]
        candidates.add(0, _search.best(block, b * size, k, candidates.floor))
    return candidates.result(min(k, len(scores)))


@pytest.mark.parametrize("k", [1, 3, 10, 64])
@pytest.mark.parametrize("seed", range(3))
def test_blocks_merged_in_any_order_give_top_k(k, seed):
    rng = np.random.default_rng(seed)
    scores = rng.integers(-3, 4, (64, 5)).astype(F32) / F32(4)  # many ties
    scores[rng.random(scores.shape) < 0.1] = -0.0  # -0.0 equals 0.0
    scores[:, 4] = 0.5  # every row ties
    top_ids, top_values = _search_blocks(scores, k, rng.permutation(4))
    for j in range(5):
        want = _reference.top_k(scores[:, j], k)
        np.testing.assert_array_equal(top_ids[j], want)
        np.testing.assert_array_equal(
            top_values[j].view(np.uint32), scores[want, j].view(np.uint32)
        )


def test_a_block_keeps_at_most_k_per_query_and_nothing_below_the_floor():
    scores = np.array([[0.1, 0.9], [0.5, 0.9], [0.3, 0.9], [0.5, 0.2]], F32)
    floor = _search.order_key(np.array([0.4, -np.inf], F32))
    queries, values, ids = _search.best(scores, 100, 2, floor)
    found = list(zip(queries.tolist(), ids.tolist(), strict=True))
    assert found == [(0, 101), (0, 103), (1, 100), (1, 101)]  # ties: the smaller rows
    np.testing.assert_array_equal(values, F32([0.5, 0.5, 0.9, 0.9]))
    assert len(_search.best(scores, 0, 2, _search.order_key(np.array([1.0, 1.0], F32)))[0]) == 0


def test_floors_are_the_kth_best_merged_so_far():
    candidates = _search.Candidates(2, 2)
    scores = np.array([[0.1, 0.9], [0.5, 0.8], [0.3, 0.7]], F32)
    candidates.add(0, _search.best(scores, 0, 2, candidates.floor))
    np.testing.assert_array_equal(candidates.floor, _search.order_key(F32([0.3, 0.8])))
    ids, values = candidates.result(2)
    assert ids.tolist() == [[1, 2], [0, 1]]
    np.testing.assert_array_equal(values, F32([[0.5, 0.3], [0.9, 0.8]]))
    with pytest.raises(RuntimeError, match="lost"):
        candidates.result(3)


def test_a_block_with_fewer_rows_than_k_keeps_all_of_them():
    queries, _, ids = _search.best(
        np.ones((3, 2), F32), 100, 10, np.full(2, _search.LOWEST, np.int32)
    )
    assert queries.tolist() == [0, 0, 0, 1, 1, 1]
    assert ids.tolist() == [100, 101, 102] * 2


def test_no_queries_and_no_rows():
    assert _search.Candidates(0, 3).result(3)[0].shape == (0, 3)
    assert _search.Candidates(2, 3).result(0)[1].shape == (2, 0)


def test_blocks_bound_the_scores_of_a_task():
    shards = [(Shard("a", 0, 100_000), {}), (Shard("b", 100_000, 10), {})]
    for queries in (1, 256, 5000):
        found = list(_search.blocks(shards, queries))
        assert found[-1] == (1, 0, 10, 100_000)
        rows = max(end - begin for _, begin, end, _ in found)
        assert rows * min(queries, 256) <= max(_search._TASK_SCORES, 4096 * 256)
        assert sum(end - begin for _, begin, end, _ in found) == 100_010
    assert [s.stop - s.start for s in _search.query_blocks(600)] == [256, 256, 88]


def test_blocks_decoded_for_blas_hold_at_most_the_decode_budget():
    shards = [(Shard("a", 0, 100_000), {})]
    for queries in (1, 9, 256):
        found = list(_search.blocks(shards, queries, decode_dim=300))
        assert max(end - begin for _, begin, end, _ in found) * 300 * 4 <= _search._DECODE_BYTES
        assert sum(end - begin for _, begin, end, _ in found) == 100_000


def test_modes():
    assert _search.is_exact("exact", 10**6)
    assert not _search.is_exact("blas", 1)
    assert _search.is_exact("auto", _search.AUTO_EXACT_MAX)
    assert not _search.is_exact("auto", _search.AUTO_EXACT_MAX + 1)
    with pytest.raises(ValueError, match="mode must be one of"):
        _search.is_exact("fast", 1)


@pytest.mark.parametrize("exact", [True, False])
def test_a_cosine_beyond_float32_raises_naming_its_rows_and_queries(exact):
    from fse._backends import get_backend

    vectors = np.full((6, 2), 3e38, F32)  # finite rows whose dot product overflows
    inverse = _reference.inverse_norms(vectors)
    shards = [
        (Shard("a", 0, 2), {"vectors": vectors[:2], "inv_norms": inverse[:2]}),
        (Shard("b", 2, 4), {"vectors": vectors[2:], "inv_norms": inverse[2:]}),
    ]
    queries = np.ones((3, 2), F32)
    with pytest.raises(ValueError, match=r"rows 0 to 1 of the store, queries 0 to 2: query 0"):
        _search.search(get_backend("cpu"), shards, queries, 2, workers=1, exact=exact)


def test_at_most_two_tasks_per_worker_are_in_flight():
    started = []

    def fn(task):
        started.append(task)
        return task

    for k, result in enumerate(_search._run(fn, list(range(100)), 3, "t")):
        assert result == k  # in task order
        assert len(started) <= k + 1 + 2 * 3
    assert len(started) == 100


def test_order_keys_follow_the_floats_and_keep_subnormals_apart():
    tiny = np.finfo(F32).smallest_subnormal
    values = F32([-np.inf, -3e38, -1.0, -tiny, -0.0, 0.0, tiny, 2 * tiny, 1e-38, 1.0, np.inf])
    keys = _search.order_key(values)
    assert keys.dtype == np.int32
    assert (np.diff(keys.astype(np.int64)) >= 0).all()
    assert keys[4] == keys[5]  # -0.0 equals 0.0
    assert len(set(keys.tolist())) == len(values) - 1  # every other pair differs
    rng = np.random.default_rng(3)
    with np.errstate(over="ignore", under="ignore"):
        scale = 10.0 ** rng.integers(-44, 38, 10_000)
        many = (rng.standard_normal(10_000) * scale).astype(F32)  # subnormals to infinities
    np.testing.assert_array_equal(
        np.argsort(_search.order_key(many), kind="stable"), np.argsort(many, kind="stable")
    )


def test_a_subnormal_score_beats_zero():
    tiny = np.finfo(F32).smallest_subnormal
    scores = np.array([[0.0], [tiny]], F32)  # row 1 is better, though only by a subnormal
    ids, _ = _search_blocks(scores, 1, [0], size=2)
    assert ids.tolist() == [[1]]
