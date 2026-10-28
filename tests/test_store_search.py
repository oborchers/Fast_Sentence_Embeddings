"""The search by blocks: the best rows of a block, merged in any order, as top_k."""

import numpy as np
import pytest

from fse import _reference
from fse._store import _search
from fse._store._layout import Shard

F32 = np.float32


@pytest.mark.parametrize("k", [1, 3, 10, 64])
def test_blocks_merged_in_any_order_give_top_k(k):
    rng = np.random.default_rng(k)
    scores = rng.integers(-3, 4, (64, 5)).astype(F32) / F32(4)  # many ties
    scores[rng.random(scores.shape) < 0.1] = -0.0  # -0.0 equals 0.0
    found = [_search.best(scores[b : b + 16], b, k) for b in range(0, 64, 16)]
    rng.shuffle(found)
    values = np.concatenate([f[0] for f in found], axis=1)
    ids = np.concatenate([f[1] for f in found], axis=1)
    top_values, top_ids = _search.merge(values, ids, k)
    for j in range(5):
        want = _reference.top_k(scores[:, j], k)
        np.testing.assert_array_equal(top_ids[j], want)
        np.testing.assert_array_equal(
            top_values[j].view(np.uint32), scores[want, j].view(np.uint32)
        )


def test_a_block_with_fewer_rows_than_k_keeps_all_of_them():
    values, ids = _search.best(np.ones((3, 2), F32), 100, 10)
    assert ids.tolist() == [[100, 101, 102], [100, 101, 102]]
    assert values.shape == (2, 3)


def test_blocks_bound_the_scores_of_a_task():
    shards = [(Shard("a", 0, 100_000), {}), (Shard("b", 100_000, 10), {})]
    for queries in (1, 256, 5000):
        found = list(_search.blocks(shards, queries))
        assert found[-1] == (1, 0, 10, 100_000)
        rows = max(end - begin for _, begin, end, _ in found)
        assert rows * min(queries, 256) <= max(_search._TASK_SCORES, 4096 * 256)
        assert sum(end - begin for _, begin, end, _ in found) == 100_010
    assert [s.stop - s.start for s in _search.query_blocks(600)] == [256, 256, 88]
