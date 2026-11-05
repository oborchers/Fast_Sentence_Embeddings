"""The best row of every document in a block, and the merge of blocks, against the reference."""

import numpy as np
import pytest

from fse import _reference
from fse._store import _search
from fse._store._unique_docs import DocumentCandidates, best_per_doc


def _best(scores, docs, k, *, sorted_docs=None, floor=None):
    """best_per_doc with a floor of -inf; returns (query, score, local row, label) tuples."""
    q = scores.shape[1]
    if sorted_docs is None:
        sorted_docs = bool(np.all(docs[:-1] <= docs[1:]))
    floor = np.full(q, _search.LOWEST, np.int32) if floor is None else floor
    return best_per_doc(scores, docs, sorted_docs, k, floor)


def _want(scores, docs, k):
    """The reference rows (positions in the block) of every query."""
    return [_reference.top_k_unique(scores[:, q], docs, k) for q in range(scores.shape[1])]


def _merged(blocks, k, queries=1):
    """Merge (scores, docs, physical) blocks; the rows, scores and labels (queries, width)."""
    out = DocumentCandidates(queries, k)
    for scores, docs, physical in blocks:
        local = _best(scores, docs, k, floor=out.floor)
        out.add(0, (local[0], local[1], physical[local[2]], local[3]))
    return out


def test_reference_top_k_unique_by_hand():
    scores = np.array([0.5, 0.9, 0.9, 0.1, 0.5], np.float32)
    labels = np.array([7, 7, 3, 3, 9])
    # 7 by row 1 (0.9, before row 2 at the tie), 3 by row 2, 9 by row 4, in the order of top_k
    assert list(_reference.top_k_unique(scores, labels, 5)) == [1, 2, 4]
    assert list(_reference.top_k_unique(scores, labels, 2)) == [1, 2]


@pytest.mark.parametrize("sorted_docs", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_block_matches_reference(seed, sorted_docs):
    rng = np.random.default_rng(seed)
    n, q, k = int(rng.integers(1, 200)), int(rng.integers(1, 6)), int(rng.integers(1, 12))
    docs = rng.integers(-3, int(rng.integers(1, 40)), n).astype(np.int64)
    if sorted_docs:
        docs.sort()
    scores = np.round(rng.standard_normal((n, q)), 1).astype(np.float32)  # many ties
    queries, values, local, labels = _best(scores, docs, k, sorted_docs=sorted_docs)
    for column, want in enumerate(_want(scores, docs, k)):
        mine = queries == column
        assert list(local[mine]) == list(want)
        assert values[mine].tobytes() == scores[want, column].tobytes()
        assert list(labels[mine]) == list(docs[want])


@pytest.mark.parametrize("seed", range(6))
def test_any_block_order_gives_the_reference(seed):
    rng = np.random.default_rng(seed)
    n, q, k = 300, 3, int(rng.integers(1, 15))
    docs = rng.integers(0, int(rng.integers(2, 60)), n).astype(np.int64)
    scores = np.round(rng.standard_normal((n, q)), 1).astype(np.float32)
    cuts = np.sort(rng.choice(np.arange(1, n), int(rng.integers(1, 8)), replace=False))
    parts = np.split(np.arange(n), cuts)
    rng.shuffle(parts)
    blocks = [(scores[p], docs[p], p.astype(np.int64)) for p in parts]
    out = DocumentCandidates(q, k)
    for scores_p, docs_p, physical in blocks:
        queries, values, local, labels = _best(scores_p, docs_p, k, floor=out.floor)
        out.add(0, (queries, values, physical[local], labels))
    rows, values, labels = out.result()
    for column, want in enumerate(_want(scores, docs, k)):
        assert list(rows[column]) == list(want)
        assert values[column].tobytes() == scores[want, column].tobytes()
        assert list(labels[column]) == list(docs[want])


def test_ties_go_to_the_smaller_row_also_at_the_cut():
    scores = np.ones((6, 1), np.float32)
    docs = np.array([4, 4, 2, 9, 2, 1])
    queries, _, local, labels = _best(scores, docs, 3, sorted_docs=False)
    assert list(local) == [0, 2, 3]  # documents 4, 2, 9 by their first rows
    assert list(labels) == [4, 2, 9]
    assert list(queries) == [0, 0, 0]


def test_negative_zero_keeps_its_bits_and_equals_zero():
    scores = np.array([[0.0], [-0.0], [-0.0], [0.0]], np.float32)
    docs = np.array([1, 1, 2, 2])
    _, values, local, _ = _best(scores, docs, 2)
    assert list(local) == [0, 2]  # -0.0 equals 0.0: the smaller row wins within document 2
    assert values.tobytes() == scores[[0, 2], 0].tobytes()
    assert np.signbit(values[1])


def test_subnormals_stay_apart_from_zero():
    tiny = np.float32(1e-45)
    scores = np.array([[0.0], [tiny], [0.0]], np.float32)
    docs = np.array([1, 1, 2])
    _, values, local, _ = _best(scores, docs, 2)
    assert list(local) == [1, 2]
    assert values[0] == tiny


def test_local_rows_are_positions_in_the_block():
    scores = np.array([[0.1], [0.9], [0.5]], np.float32)
    physical = np.array([2, 9, 40])
    _, _, local, _ = _best(scores, np.array([5, 5, 6]), 2)
    assert local.dtype == np.int32
    assert list(physical[local]) == [9, 40]


def test_a_block_keeps_at_most_k_groups_per_query():
    scores = np.ones((50, 4), np.float32)
    docs = np.arange(50)
    queries, _, local, _ = _best(scores, docs, 5)
    assert list(np.bincount(queries)) == [5] * 4
    assert set(local) == {0, 1, 2, 3, 4}


def test_floor_is_the_kth_distinct_document_and_rows_below_it_vanish():
    out = DocumentCandidates(1, 2)
    assert out.floor[0] == _search.LOWEST
    two_rows = (
        np.zeros(2, np.int64),
        np.array([0.9, 0.8], np.float32),
        np.array([0, 1]),
        np.array([1, 1]),
    )
    out.add(0, two_rows)
    assert out.floor[0] == _search.LOWEST  # two rows of one document are one document, not k
    out.add(0, (np.zeros(1, np.int64), np.array([0.7], np.float32), np.array([2]), np.array([2])))
    out.add(0, (np.zeros(1, np.int64), np.array([0.5], np.float32), np.array([3]), np.array([3])))
    assert _search.from_key(out.floor)[0] == np.float32(0.7)
    scores = np.array([[0.69], [0.2]], np.float32)
    assert len(_best(scores, np.array([5, 6]), 2, floor=out.floor)[0]) == 0  # all below: empty


def test_a_later_better_row_replaces_the_representative():
    blocks = [
        (np.array([[0.5], [0.4]], np.float32), np.array([8, 9]), np.array([0, 1])),
        (np.array([[0.7]], np.float32), np.array([8]), np.array([5])),
    ]
    rows, values, labels = _merged(blocks, 2).result()
    assert list(rows[0]) == [5, 1]
    assert list(labels[0]) == [8, 9]
    assert values[0, 0] == np.float32(0.7)


def test_an_equal_later_row_keeps_the_earlier_smaller_position():
    blocks = [
        (np.array([[0.5]], np.float32), np.array([8]), np.array([3])),
        (np.array([[0.5]], np.float32), np.array([8]), np.array([7])),
    ]
    assert list(_merged(blocks, 1).result()[0][0]) == [3]


def test_one_document_is_one_group_and_few_groups_are_all_kept():
    scores = np.arange(10, dtype=np.float32).reshape(10, 1)
    _, _, local, labels = _best(scores, np.full(10, 3), 4)
    assert list(local) == [9]
    assert list(labels) == [3]
    assert len(_best(scores, np.arange(10), 20)[0]) == 10


def test_a_lost_block_is_an_error():
    out = DocumentCandidates(2, 1)
    out.add(0, (np.zeros(1, np.int64), np.ones(1, np.float32), np.array([0]), np.array([1])))
    with pytest.raises(RuntimeError, match="lost"):
        out.result()
