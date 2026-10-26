#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""The bitmap that tracks repeated indices in the scan, in Python and compiled."""

import random
import re
import tracemalloc

import pytest

from fse.models import _index_tracker, base_s2v
from fse.models._index_tracker import MAX_INDEX, IndexTracker
from fse.models.average_inner import scan_indexed_sentences
from fse.models.base_s2v import BaseSentence2VecModel


@pytest.fixture
def w2v(make_w2v):
    return make_w2v(10)


def _reference(indices):
    """Whether every index occurs at most once, with a set."""
    return len(set(indices)) == len(indices)


def _tracked(indices):
    """The Python tracker as the scan loop drives it."""
    tracker = None
    for i, index in enumerate(indices):
        if tracker is None and index != i:
            tracker = IndexTracker(i)
        if tracker is not None and not tracker.add(index, i + 1):
            return False
    return tracker is None or tracker.unique(len(indices))


def _compiled(indices):
    items = [(["a"], index) for index in indices]
    return scan_indexed_sentences(items, 1e9, lambda *args: None)[4]


def _cases():
    rng = random.Random(3)
    n = 3000
    permutation = list(range(n))
    rng.shuffle(permutation)
    sparse = rng.sample(range(10**9), n)
    cases = [
        [],
        [0],
        [5],
        list(range(n)),
        list(reversed(range(n))),
        permutation,
        [*permutation[:-1], permutation[0]],
        [permutation[1], *permutation[1:]],
        [*range(100), 10**7, *range(100, 200)],
        sparse,
        [*sparse, sparse[17]],
        [sparse[-1], *sparse],
        [*range(50), 49],
        [MAX_INDEX, 0, MAX_INDEX],
        [MAX_INDEX, 0, 1],
    ]
    for _ in range(20):
        size = rng.randrange(1, 400)
        cases.append([rng.randrange(size * rng.choice([1, 2, 1000])) for _ in range(size)])
    return cases


@pytest.mark.parametrize("min_bytes", [1, 64, _index_tracker.BITMAP_MIN_BYTES])
@pytest.mark.parametrize("indices", _cases())
def test_both_trackers_match_a_set(monkeypatch, min_bytes, indices):
    # A small minimum exercises the pending indices, the folds and the final sort.
    monkeypatch.setattr(_index_tracker, "BITMAP_MIN_BYTES", min_bytes)
    expected = _reference(indices)
    assert _tracked(indices) is expected
    assert bool(_compiled(indices)) is expected


@pytest.mark.parametrize("indices", [[1, 0, 2, 3], [3, 1, 0, 2], [0, 2, 1, 1]])
def test_both_scan_paths_agree(w2v, monkeypatch, indices):
    items = [(["the"], index) for index in indices]
    compiled = BaseSentence2VecModel(w2v).scan_sentences(items)
    monkeypatch.setattr(base_s2v, "scan_indexed_sentences", None)
    assert BaseSentence2VecModel(w2v).scan_sentences(items) == compiled
    assert compiled["unique_indices"] is _reference(indices)


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize(
    ("indices", "position"),
    [
        ([0, MAX_INDEX + 1], 1),
        ([MAX_INDEX + 1, 0], 0),
        # After a repeat the tracker is gone; the bound still applies.
        ([0, 0, MAX_INDEX + 1], 2),
        ([1, 1, 2**70], 2),
    ],
)
def test_index_beyond_int64_raises(w2v, monkeypatch, compiled, indices, position):
    if not compiled:
        monkeypatch.setattr(base_s2v, "scan_indexed_sentences", None)
    items = [(["the"], index) for index in indices]
    message = (
        f"At {position}: Passed index {indices[position]}, larger than the largest "
        f"supported index {MAX_INDEX}"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        BaseSentence2VecModel(w2v).scan_sentences(items)


def test_memory_of_a_permutation_is_bounded(monkeypatch):
    """A permutation of 10^6 indices takes a few megabytes, a set would take about 60."""
    n = 10**6
    rng = random.Random(0)
    indices = list(range(n))
    rng.shuffle(indices)
    monkeypatch.setattr(_index_tracker, "BITMAP_MIN_BYTES", 1024)
    # Count only what the tracker allocates, also if tracemalloc already runs.
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        assert _tracked(indices)
        _, peak = tracemalloc.get_traced_memory()
        peak -= before
    finally:
        if started:
            tracemalloc.stop()
    # About one byte per sentence waits on the way; folding it takes a few temporaries of
    # that size, and the final bitmap has n / 8 bytes.
    assert peak < 6 * n
