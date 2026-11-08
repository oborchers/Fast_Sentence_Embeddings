"""Stores with sign bits and the oracle of the binary search: the candidates of the reference
(``fse._reference_bits.candidates``) and the exact search of exactly those rows."""

import shutil

import numpy as np

import fse
from fse import _reference_bits
from support import rescore as rs

F32 = np.float32


def make(path, dtype, n=60, *, shard_rows=7, docs=None, bits=True, seed=0):
    """A store like ``rescore.make``; with ``bits`` every shard has its artifact."""
    codes = rs.make(path, dtype, n, shard_rows=shard_rows, docs=docs, seed=seed)
    if bits:
        fse.add_sign_bits(path)
    return codes


def drop_bits(path, *first_rows):
    """Remove the artifacts of the shards starting at ``first_rows``."""
    for first in first_rows:
        shutil.rmtree(path / f"sign_bits-{first:016d}")


def candidates(codes, qs, m, selection=None):
    """Per query the sorted candidate positions the reference chooses among ``selection``."""
    rows = None if selection is None else np.asarray(selection, np.int64)
    stored = codes if rows is None else codes[rows]
    found = _reference_bits.candidates(
        _reference_bits.sign_bits(stored), _reference_bits.query_bits(qs), m, rows
    )
    return [np.sort(c) for c in found]


def assert_binary_is_exact_on_candidates(store, qs, k, m, selection=None, codes=None, **kwargs):
    """``search(mode="binary")`` equals ``search(mode="exact", rows=sorted(C))`` per query."""
    got = store.search(qs, k, mode="binary", oversample=-(-m // k), **kwargs)
    own = {name: value for name, value in kwargs.items() if name != "rows"}
    want = []
    for q, cand in enumerate(candidates(codes, qs, -(-m // k) * k, selection)):
        want.append(store.search(qs[q], k, mode="exact", rows=cand.tolist(), **own))
    width = max(len(w.ids) for w in want)
    assert got.ids.shape == (len(qs), width)
    for q, w in enumerate(want):
        t = len(w.ids)
        np.testing.assert_array_equal(got.ids[q, :t], w.ids)
        np.testing.assert_array_equal(rs.bits(got.scores[q, :t]), rs.bits(w.scores))
        assert (got.ids[q, t:] == -1).all()
        if hasattr(w, "doc_ids"):
            np.testing.assert_array_equal(got.doc_ids[q, :t], w.doc_ids)
    return got
