"""``fse._store._vectorstore_search``: the bodies of ``VectorStore.search`` and ``rescore``."""

from __future__ import annotations

import numpy as np
import pydantic
import pytest

import fse
from fse import VectorStore
from fse._store import _vectorstore_search
from fse._validation import PositiveInt, validate_arguments
from support.encoder import finished
from support.runtime import texts

ENCODER = finished()


def _write(path, sentences, **options):
    with fse.ShardSink(path, ENCODER, shard_rows=16, **options) as sink:
        if options.get("doc_ids"):
            ENCODER.encode(sentences, sink=sink, doc_ids=[i // 3 for i in range(len(sentences))])
        else:
            ENCODER.encode(sentences, sink=sink)
    return path


def _queries(n):
    return np.random.default_rng(n).standard_normal((n, ENCODER.output_dim)).astype(np.float32)


def _same(a, b):
    assert type(a) is type(b)
    np.testing.assert_array_equal(a.ids, b.ids)
    np.testing.assert_array_equal(a.scores.view(np.uint32), b.scores.view(np.uint32))
    if hasattr(a, "doc_ids"):
        np.testing.assert_array_equal(a.doc_ids, b.doc_ids)


@pytest.mark.parametrize("n", [1, 3])
def test_search_store_is_what_the_method_gives(tmp_path, n):
    path = _write(tmp_path / "a", texts(60))
    queries = _queries(n)[0] if n == 1 else _queries(n)
    with VectorStore.open(path, encoder=ENCODER) as store:
        direct = _vectorstore_search.search_store(
            store, queries, 7, workers=2, mode="exact", doc_ids=None, rows=None, max_per_doc=None
        )
        _same(direct, store.search(queries, 7, workers=2, mode="exact"))


def test_search_store_selects_rows(tmp_path):
    path = _write(tmp_path / "a", texts(60))
    with VectorStore.open(path, encoder=ENCODER) as store:
        direct = _vectorstore_search.search_store(
            store,
            _queries(2),
            5,
            workers=None,
            mode="exact",
            doc_ids=None,
            rows=[3, 9, 20, 41],
            max_per_doc=None,
        )
        _same(direct, store.search(_queries(2), 5, mode="exact", rows=[3, 9, 20, 41]))
        assert set(direct.ids.ravel()) <= {3, 9, 20, 41}


def test_search_store_groups_by_document(tmp_path):
    path = _write(tmp_path / "a", texts(60), doc_ids=True)
    with VectorStore.open(path, encoder=ENCODER) as store:
        direct = _vectorstore_search.search_store(
            store,
            _queries(1)[0],
            5,
            workers=1,
            mode="exact",
            doc_ids=None,
            rows=None,
            max_per_doc=1,
        )
        _same(direct, store.search(_queries(1)[0], 5, workers=1, mode="exact", max_per_doc=1))


def test_rescore_store_is_what_the_method_gives(tmp_path):
    path = _write(tmp_path / "a", texts(60))
    candidates = np.array([[5, 1, 33, -1, 1], [50, 2, -1, -1, -1]], dtype=np.int64)
    with VectorStore.open(path, encoder=ENCODER) as store:
        direct = _vectorstore_search.rescore_store(
            store, _queries(2), candidates, 3, workers=2, max_per_doc=None
        )
        _same(direct, store.rescore(_queries(2), candidates, 3, workers=2))


def test_rescore_store_equals_an_exact_search_of_the_rows(tmp_path):
    path = _write(tmp_path / "a", texts(60))
    with VectorStore.open(path, encoder=ENCODER) as store:
        got = _vectorstore_search.rescore_store(
            store, _queries(1)[0], [7, 3, 7, 12], 3, workers=None, max_per_doc=None
        )
        _same(got, store.search(_queries(1)[0], 3, mode="exact", rows=[3, 7, 12]))


def test_a_closed_store_raises_in_both(tmp_path):
    path = _write(tmp_path / "a", texts(20))
    store = VectorStore.open(path, encoder=ENCODER)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        _vectorstore_search.search_store(
            store,
            _queries(1),
            3,
            workers=None,
            mode="auto",
            doc_ids=None,
            rows=None,
            max_per_doc=None,
        )
    with pytest.raises(RuntimeError, match="closed"):
        _vectorstore_search.rescore_store(
            store, _queries(1), [[1, 2]], 3, workers=None, max_per_doc=None
        )


def _reference_title(name):
    """The title pydantic gives a method named ``VectorStore.<name>`` decorated like the real one.

    pydantic before 2.6 titles from ``__name__`` ("search"), later versions from
    ``__qualname__`` ("VectorStore.search"), so the rule is read off a stand-in instead of
    spelled out.
    """

    def method(self, queries, k: PositiveInt = 10):
        raise AssertionError(("not reached", self, queries, k))

    method.__name__ = name
    method.__qualname__ = f"VectorStore.{name}"
    stand_in = type("StandIn", (), {name: validate_arguments(method)})
    with pytest.raises(pydantic.ValidationError) as caught:
        getattr(stand_in(), name)(None, 0)
    return caught.value.title


@pytest.mark.parametrize("name", ["search", "rescore"])
def test_validation_errors_are_titled_with_the_method(tmp_path, name):
    expected = _reference_title(name)
    assert expected.endswith(name)
    path = _write(tmp_path / "a", texts(20))
    with VectorStore.open(path, encoder=ENCODER) as store:
        call = store.search if name == "search" else lambda q, k: store.rescore(q, [[1]], k)
        with pytest.raises(pydantic.ValidationError) as caught:
            call(_queries(1), 0)
        assert caught.value.title == expected
