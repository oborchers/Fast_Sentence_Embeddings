"""``VectorStore.rescore`` on views: the bits of one store that holds the same rows."""

import numpy as np
import pytest

import fse
from support import rescore as rs
from support.views import vector_parts


@pytest.mark.parametrize("dtype", ["float32", "int8"])
@pytest.mark.parametrize("grouped", [False, True])
def test_a_view_gives_what_one_store_gives(tmp_path, dtype, grouped):
    whole, parts = vector_parts(tmp_path, dtype=dtype)
    with fse.VectorStore.open(whole) as one, fse.VectorStore.open_view(parts) as view:
        n = len(one)
        assert len(view) == n
        qs = np.random.default_rng(4).standard_normal((5, one.dim)).astype(np.float32)
        cands = rs.candidate_lists(n, 5, width=25, seed=8)
        arguments = {"max_per_doc": 1} if grouped else {}
        a = one.rescore(qs, cands, 6, **arguments)
        b = view.rescore(qs, cands, 6, workers=3, **arguments)
        np.testing.assert_array_equal(a.ids, b.ids)
        np.testing.assert_array_equal(rs.bits(a.scores), rs.bits(b.scores))
        np.testing.assert_array_equal(a.doc_ids, b.doc_ids)
        # repeated sentences give equal rows across the members: ties by position
        everything = np.tile(np.arange(n, dtype=np.int64), (5, 1))
        c = view.rescore(qs, everything, 9)
        d = view.search(qs, 9, mode="exact")
        np.testing.assert_array_equal(c.ids, d.ids)
        np.testing.assert_array_equal(rs.bits(c.scores), rs.bits(d.scores))
