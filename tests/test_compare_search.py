"""Index recall must distinguish tied neighbors from missed neighbors."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def search_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("compare_search")


def test_boundary_ties_do_not_hide_canonical_id_disagreement(search_module):
    matrix = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], np.float32)
    ids = np.array([[1, 2], [1, 2], [1, 2], [3, 2]])
    found = search_module.index_recall(matrix, ids, 2)
    assert found["recall_at_k"] == 0.5
    assert found["boundary_equivalent_recall_at_k"] == 1
    assert found["queries_with_boundary_ties"] == 4


def test_wrong_neighbors_are_not_accepted_as_ties(search_module):
    matrix = np.eye(3, dtype=np.float32)
    ids = np.array([[1], [2], [0]])
    found = search_module.index_recall(matrix, ids, 1)
    assert found["recall_at_k"] == 0
    assert found["boundary_equivalent_recall_at_k"] == 0


@pytest.mark.parametrize(
    "ids", [np.array([[0, 0], [0, 1], [0, 1]]), np.array([[0, 3], [0, 1], [0, 1]])]
)
def test_invalid_ids_cannot_inflate_recall(search_module, ids):
    with pytest.raises(ValueError, match=r"duplicate|outside"):
        search_module.index_recall(np.eye(3, dtype=np.float32), ids, 2)


def test_zero_queries_are_visible_and_excluded_from_quality(search_module):
    matrix = np.array([[1, 0], [0, 1], [0, 0]], np.float32)
    found = search_module.index_recall(matrix, np.array([[0], [1], [2]]), 1)
    assert found["recall_at_k"] == 1
    assert found["zero_queries_excluded_from_recall"] == 1
    assert found["query_rows"] == 2


def test_fse_store_preserves_shared_vectors(search_module, tmp_path):
    from fse import Encoder

    matrix = np.eye(3, dtype=np.float32)
    encoder = Encoder.from_arrays(matrix, words=["a", "b", "c"])
    path = tmp_path / "index"
    index = search_module.make_index("fse", matrix, encoder, path, 1)
    search_module.save_index("fse", index, path)
    index = search_module.open_index("fse", path, 3, 1)
    try:
        assert np.array_equal(index.get_vectors(), matrix)
        ids = search_module.search("fse", index, matrix, 1, 1)
        assert np.array_equal(ids, [[0], [1], [2]])
    finally:
        index.close()


def test_boundary_matches_cannot_replace_a_strictly_better_neighbor(search_module):
    matrix = np.array([[1, 0], [0, 1], [0, -1]], np.float32)
    ids = np.array([[1, 2], [1, 0], [2, 0]])
    found = search_module.index_recall(matrix, ids, 2)
    assert found["boundary_equivalent_recall_at_k"] == 5 / 6
