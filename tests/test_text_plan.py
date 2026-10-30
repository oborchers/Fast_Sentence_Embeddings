"""Tests of fse._text._plan: chunks of about equal bytes from the offsets of sentences."""

import time
from itertools import pairwise

import numpy as np
import pytest

from fse._text import _plan
from fse._text._plan import offset_cuts


def _offsets(lengths, base=0):
    out = np.zeros(len(lengths) + 1, np.int64)
    np.cumsum(lengths, out=out[1:])
    return out + base


def _check(offsets, chunk_bytes, chunk_rows):
    cuts = offset_cuts(offsets, chunk_bytes, chunk_rows)
    n = len(offsets) - 1
    assert cuts[0] == 0
    assert cuts[-1] == n
    assert (np.diff(cuts) > 0).all()  # in order, no empty chunk
    for start, end in pairwise(cuts):
        assert end - start <= chunk_rows
        last = offsets[end] - offsets[end - 1]
        assert offsets[end] - offsets[start] <= chunk_bytes + last
    total = int(offsets[-1] - offsets[0])
    pieces = max(1, -(-total // chunk_bytes))
    assert len(cuts) - 1 <= pieces + -(-n // chunk_rows)
    return cuts


@pytest.mark.parametrize("seed", range(40))
def test_random_sources_keep_every_bound(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 3000))
    lengths = rng.integers(0, 200, n)
    lengths[rng.random(n) < 0.2] = 0  # runs of empty sentences
    if seed % 3 == 0:
        lengths[int(rng.integers(n))] = int(rng.integers(1, 1 << 21))  # one giant sentence
    chunk_bytes = int(rng.choice([1, 7, 100, 4096, 1 << 16, 1 << 21]))
    chunk_rows = int(rng.choice([1, 3, 50, 1 << 16]))
    _check(_offsets(lengths, base=int(rng.integers(0, 1000))), chunk_bytes, chunk_rows)


def test_edges():
    assert offset_cuts(np.zeros(1, np.int64), 10, 10).tolist() == [0]
    assert offset_cuts(_offsets([0] * 10), 10, 3).tolist() == [0, 2, 5, 7, 10]  # rows alone
    assert offset_cuts(_offsets([5]), 1, 1).tolist() == [0, 1]
    assert offset_cuts(_offsets([10] * 10), 50, 100).tolist() == [0, 5, 10]  # an exact multiple
    assert offset_cuts(_offsets([10] * 10, base=7), 50, 100).tolist() == [0, 5, 10]
    assert offset_cuts(_offsets([10] * 9), 30, 100).tolist() == [0, 3, 6, 9]


def test_the_chunks_are_balanced():
    cuts = offset_cuts(_offsets([100] * 1000), 25_000, 1 << 16)
    assert np.diff(cuts).tolist() == [250] * 4
    rows = offset_cuts(_offsets([0] * 10), 1, 4)  # rows split evenly, not 4, 4, 2
    assert np.diff(rows).tolist() == [3, 3, 4]


def test_the_work_is_bounded_by_the_cuts(monkeypatch):
    calls = []
    real = np.searchsorted

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_plan.np, "searchsorted", counted)
    start = time.perf_counter()
    assert offset_cuts(_offsets([1 << 20, 1 << 20]), 1, 1 << 16).tolist() == [0, 1, 2]
    assert len(calls) <= 3  # two million targets, one step per cut
    calls.clear()
    cuts = offset_cuts(_offsets([3] * 300_000), 1, 1 << 16)
    assert len(cuts) == 300_001
    assert len(calls) <= 300_001
    assert time.perf_counter() - start < 30
