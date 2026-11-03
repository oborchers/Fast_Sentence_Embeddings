"""Filtered scores match independent exact evaluation of every eligible physical row."""

from types import SimpleNamespace

import numpy as np
import pytest

import fse
from fse import _reference
from support.conformance import BACKENDS, assert_bits_equal
from support.document_ids import column, write_store


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_filtered_conformance(tmp_path, dtype, ordered, backend):
    labels = np.array([5, 1, 5, -1, 0, 5, 2, 0, 2, 8, 5, 5], dtype="int64")
    if ordered:
        labels.sort()
    write_store(tmp_path / "s", labels, dtype=dtype, shard_rows=4)
    queries = np.random.default_rng(1).standard_normal((3, 17)).astype("float32")
    positions = np.flatnonzero(np.isin(labels, [5, -1]))
    with fse.VectorStore.open(tmp_path / "s") as store:
        store._backend = SimpleNamespace(float32=store._backend.float32, cosines=backend.cosines)
        values = _reference.cosines(
            column(store, "vectors")[positions], column(store, "inv_norms")[positions], queries
        )
        order = np.stack([_reference.top_k(values[:, q], 4) for q in range(3)])
        want_ids = positions[order]
        want_scores = np.stack([values[order[q], q] for q in range(3)])
        for kwargs in ({"doc_ids": [5, -1, 5]}, {"rows": positions[::-1].copy()}):
            got = store.search(queries, 4, mode="exact", workers=2, **kwargs)
            assert np.array_equal(got.ids, want_ids)
            assert_bits_equal(got.scores, want_scores)
            assert np.array_equal(got.doc_ids, labels[want_ids])
            assert len(got) == 2
        assert store.search(queries[0], 100, doc_ids=[]).ids.shape == (0,)
        assert len(store.search(queries[0], 100, doc_ids=[5]).ids) == np.count_nonzero(labels == 5)


def test_filters_and_result_survive_close(tmp_path):
    encoder = write_store(tmp_path / "s", [5, 5, 5, 5], shard_rows=2)
    with fse.VectorStore.open(tmp_path / "s") as store:
        query = encoder.encode(["w1"])[0]
        a = store.search(query, 3, mode="exact")
        b = store.search(query, 3, rows=[0, 1, 2, 3], mode="exact")
        assert np.array_equal(a.ids, b.ids)
        assert a.scores.tobytes() == b.scores.tobytes()
        with pytest.raises(ValueError, match="either"):
            store.search(query, doc_ids=[], rows=[])
        with pytest.raises(ValueError, match="positions"):
            store.search(query, rows=[4])
        with pytest.raises(ValueError, match="zero"):
            store.search(np.zeros(17), doc_ids=[])
    assert list(a.doc_ids) == [5, 5, 5]
