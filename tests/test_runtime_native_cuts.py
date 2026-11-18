"""Native refinement preserves coverage and bounds planning memory before execution."""

from itertools import pairwise

import numpy as np
import pytest

from fse._runtime._native_cuts import refine
from fse._runtime._tasks import Task
from fse._text import Chunk, StringSource, _plan, copied_strings


@pytest.mark.parametrize("source_factory", [StringSource, copied_strings])
@pytest.mark.parametrize("workers", [2, 4, 8, 2**64])
@pytest.mark.parametrize("seed", range(12))
def test_cut_properties(source_factory, workers, seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(0, 100, 1000)
    sizes[rng.random(1000) < 0.3] = 0
    if seed % 3 == 0:
        sizes[450] = 70000
    source = source_factory([" " * int(n) for n in sizes])
    boundaries = [0, 70, 500, 600, 1000]
    tasks = [Task(source, Chunk(a, b, a, b - a)) for a, b in pairwise(boundaries)]
    result = refine(source, tasks, workers, 256)
    assert result == refine(source, tasks, workers, 256)
    if result is None:
        return
    cuts = [result[0].chunk.start, *(task.chunk.end for task in result)]
    assert cuts[0] == 0
    assert cuts[-1] == len(source)
    assert set(boundaries) <= set(cuts)
    target = max(1, -(-source.nbytes // (4 * workers)))
    for task in result:
        assert task.source is source
        assert task.chunk.start == task.first_row
        assert task.chunk.end - task.chunk.start == task.n_rows > 0
        assert any(a <= task.chunk.start < task.chunk.end <= b for a, b in pairwise(boundaries))
        size = int(source._offsets[task.chunk.end] - source._offsets[task.chunk.start])
        last = int(source._offsets[task.chunk.end] - source._offsets[task.chunk.end - 1])
        assert size <= target + last
    assert all(a < b for a, b in pairwise(cuts))


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_uniform_source_has_four_to_eight_items_per_worker(workers):
    source = StringSource(["w0 " * 100] * 4000)
    from support.native_team import pipeline, tasks_for

    original = tasks_for(pipeline(), source, workers, refined=False)
    found = refine(source, original, workers, 256)
    assert 4 * workers <= len(found) <= 8 * workers
    assert len(found) > len(original)


@pytest.mark.parametrize("sizes", [[0] * 50, [70000], [0] * 50 + [70000] + [0] * 50])
def test_empty_ranges_and_indivisible_sentences(sizes):
    source = StringSource([" " * n for n in sizes])
    task = Task(source, Chunk(0, len(source), 0, len(source)))
    result = refine(source, [task], 2**64, 256)
    assert sum(part.n_rows for part in result) == len(source)
    assert all(part.n_rows > 0 for part in result)
    assert len(result) <= 3


@pytest.mark.parametrize("count", [255, 256, 257, 131072])
def test_cut_build_stops_before_creating_excess_tasks(monkeypatch, count):
    source = StringSource(["x"] * count)
    task = Task(source, Chunk(0, count, 0, count))
    searches = []
    real = _plan.np.searchsorted

    def search(*args, **kwargs):
        searches.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_plan.np, "searchsorted", search)
    found = refine(source, [task], 2**64, 256)
    assert len(searches) <= 257
    if count <= 256:
        assert len(found) == count
    else:
        assert found is None
