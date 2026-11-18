"""The per-call StringSource team keeps the reference bits and ordered fallback."""

import numpy as np
import pytest

from fse import _native
from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline, _native_run
from fse._text import StringSource
from fse._tokenizer_spec import TokenizerSpec
from support.runtime import expected, texts, word_vectors


def pipeline(normalize=True):
    wv = word_vectors()
    return Pipeline(
        EncodeSpec(
            wv,
            np.ones(len(wv), np.float32),
            (Normalize(),) if normalize else (),
            TokenizerSpec.of("fse_words@1"),
        )
    )


@pytest.mark.parametrize("workers", [2, 3, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_team_matches_reference_and_old_executor(monkeypatch, workers, normalize):
    model = pipeline(normalize)
    sentences = texts(2000)
    source = StringSource(sentences)
    calls = _native._test_team_calls()
    found = model.infer(source, workers=workers)
    assert _native._test_team_calls() == calls + 1
    assert _native._test_team_resources() == 0
    np.testing.assert_array_equal(
        found.view(np.uint32), expected(model.spec, sentences).view(np.uint32)
    )
    monkeypatch.setattr(_native_run, "eligible", lambda *_args: False)
    old = model.infer(source, workers=workers)
    np.testing.assert_array_equal(found.view(np.uint32), old.view(np.uint32))


@pytest.mark.parametrize("failed_starts", [0, 2, 10, 254])
def test_failed_thread_starts_still_drain_the_queue(failed_starts):
    model = pipeline()
    sentences = texts(2000)
    source = StringSource(sentences)
    previous = _native._test_team_faults(failed_starts=failed_starts)
    try:
        calls = _native._test_team_calls()
        found = model.infer(source, workers=8)
        assert _native._test_team_calls() == calls + 1
        np.testing.assert_array_equal(
            found.view(np.uint32), expected(model.spec, sentences).view(np.uint32)
        )
        assert _native._test_team_resources() == 0
    finally:
        _native._test_team_faults(*previous)


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (1, MemoryError, "could not grow the buffers of the Workspace"),
        (2, RuntimeError, "native task team and Python task error mismatch"),
    ],
)
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_failure_status_preserves_prefix_and_replay_contract(
    monkeypatch, status, error_type, message, position
):
    from fse._runtime import _pipeline
    from support.native_team import faults, tasks_for

    model = pipeline()
    source = StringSource(texts(4000))
    tasks = tasks_for(model, source, 4)
    index = {"first": 0, "middle": len(tasks) // 2, "last": len(tasks) - 1}[position]
    allocations, replays = [], []
    real_own, real_work = _pipeline.own_rows, Pipeline._task_work

    def own(*args):
        rows = real_own(*args)
        allocations.append(rows)
        return rows

    def work_factory(*args):
        work = real_work(*args)

        def replay(item):
            replays.append(item[0])
            return work(item)

        return replay

    monkeypatch.setattr(_pipeline, "own_rows", own)
    monkeypatch.setattr(Pipeline, "_task_work", work_factory)
    before = _native._test_team_calls()
    with faults(status_task=index, status=status), pytest.raises(error_type, match=message):
        model.infer(source, workers=4)
    assert _native._test_team_calls() == before + 1
    assert allocations[0].next_row == sum(task.n_rows for task in tasks[:index])
    assert len(replays) == (0 if status == 1 else 1)
    assert _native._test_team_resources() == 0


def _fixed_plan(monkeypatch, count, *, scan_error=False):
    import threading

    from fse._runtime._tasks import Plan, Task
    from fse._text import Chunk

    source = StringSource(["w0 " + " " * 300] * count)
    seen = []

    def items():
        for i in range(count):
            seen.append(i)
            yield Task(source, Chunk(i, i + 1, i, 1))
        if scan_error:
            error_type = MemoryError if scan_error == "memory" else ValueError
            raise error_type("scan stopped after complete tasks")

    def plan(*_args, **_kwargs):
        return Plan(items(), count, False), 8, threading.Event()

    monkeypatch.setattr(Pipeline, "plan_tasks", plan)
    return source, seen


@pytest.mark.parametrize("count", [255, 256, 257, 258])
def test_task_limit_buffers_at_most_257_then_replays_all_tasks(monkeypatch, count):
    source, seen = _fixed_plan(monkeypatch, count)
    observed = []
    real = _native_run._buffer

    def buffer(remaining):
        result = real(remaining)
        observed.append(len(seen))
        return result

    monkeypatch.setattr(_native_run, "_buffer", buffer)
    before = _native._test_team_calls()
    found = pipeline().infer(source, workers=4)
    assert len(found) == count
    assert observed == [min(count, 257)]
    assert seen == list(range(count))
    assert _native._test_team_calls() == before + (count <= 256)


@pytest.mark.parametrize("count", [256, 257])
@pytest.mark.parametrize("preparation_failure", [False, True])
def test_scan_error_is_preserved_by_team_and_fallback(monkeypatch, count, preparation_failure):
    from support.native_team import faults

    source, seen = _fixed_plan(monkeypatch, count, scan_error=True)
    before = _native._test_team_calls()
    with (
        faults(allocation=int(preparation_failure)),
        pytest.raises(ValueError, match="scan stopped"),
    ):
        pipeline().infer(source, workers=4)
    assert seen == list(range(count))
    assert _native._test_team_calls() == before + (count <= 256 and not preparation_failure)
    assert _native._test_team_resources() == 0


def test_append_failure_keeps_the_item_and_remaining_iterator(monkeypatch):
    source, seen = _fixed_plan(monkeypatch, 258)
    real = _native_run._append
    attempts = []

    def append(buffered, item):
        attempts.append(1)
        if len(attempts) == 7:
            raise MemoryError("could not buffer a task")
        real(buffered, item)

    monkeypatch.setattr(_native_run, "_append", append)
    before = _native._test_team_calls()
    found = pipeline().infer(source, workers=4)
    assert len(found) == 258
    assert seen == list(range(258))
    assert len(attempts) == 7
    assert _native._test_team_calls() == before


@pytest.mark.parametrize("difference", [-1, 0, 1])
def test_actual_row_limit_routes_below_at_and_above_boundary(difference):
    count = _native_run.MAX_ROWS + difference
    source = StringSource(["w0"] * count)
    before = _native._test_team_calls()
    found = pipeline().infer(source, workers=2)
    assert found.shape == (count, 12)
    assert _native._test_team_calls() == before + (difference <= 0)


@pytest.mark.parametrize("difference", [-1, 0, 1])
def test_actual_byte_limit_routes_below_at_and_above_boundary(difference):
    count = _native_run.MAX_BYTES + difference
    first, remainder = divmod(count, 256)
    source = StringSource(["w0" + " " * (first - 2)] * 255 + ["w0" + " " * (first + remainder - 2)])
    assert source.nbytes == count
    before = _native._test_team_calls()
    found = pipeline().infer(source, workers=2)
    assert found.shape == (256, 12)
    assert _native._test_team_calls() == before + (difference <= 0)


@pytest.mark.parametrize("failed", [False, True])
def test_native_plan_is_logged_once_and_errors_have_no_done_record(failed):
    import threading

    from support.logs import captured
    from support.native_team import faults, tasks_for

    model = pipeline()
    source = StringSource(texts(2000))
    tasks = tasks_for(model, source, 2)
    with captured() as records, faults(status_task=0 if failed else -1, status=1):
        if failed:
            with pytest.raises(MemoryError):
                model.infer(source, workers=2)
        else:
            model.infer(source, workers=2)
    events = [record for record in records if record.fse_event in ("runtime.plan", "runtime.done")]
    assert [record.fse_event for record in events] == (
        ["runtime.plan"] if failed else ["runtime.plan", "runtime.done"]
    )
    assert events[0].fse_executor == "native"
    assert all(record.threadName == threading.current_thread().name for record in events)
    if not failed:
        assert (events[1].fse_rows, events[1].fse_tasks) == (len(source), len(tasks))


@pytest.mark.parametrize("candidate", [False, True])
def test_setup_allocation_failure_keeps_one_plan_record(monkeypatch, candidate):
    from fse._runtime import _pipeline
    from support.logs import captured

    model = pipeline()
    source = StringSource(texts(2000))
    with captured() as records:

        def fail(*_args):
            if not candidate:
                assert [r.fse_event for r in records] == ["runtime.plan"]
            raise MemoryError("own output allocation failed")

        monkeypatch.setattr(_pipeline, "own_rows", fail)
        with pytest.raises(MemoryError, match="own output allocation failed"):
            model.infer(source, workers=2, chunk_bytes=None if candidate else 8000)
    assert [r.fse_event for r in records] == ["runtime.plan"]


def test_next_memory_error_is_a_scan_error_not_a_buffering_failure(monkeypatch):
    source, seen = _fixed_plan(monkeypatch, 256, scan_error="memory")
    before = _native._test_team_calls()
    with pytest.raises(MemoryError, match="scan stopped after complete tasks"):
        pipeline().infer(source, workers=2)
    assert seen == list(range(256))
    assert _native._test_team_calls() == before + 1


def test_native_range_invariant_is_not_hidden_by_fallback(monkeypatch):
    from dataclasses import replace

    from fse._runtime import _pipeline
    from fse._runtime._tasks import Plan
    from support.native_team import tasks_for

    model = pipeline()
    source = StringSource(texts(2000))
    tasks = tasks_for(model, source, 2)
    assert len(tasks) > 1
    tasks[1] = replace(tasks[1], chunk=replace(tasks[1].chunk, first_row=tasks[1].first_row + 1))
    original_plan, window, stop = model.plan_tasks(source, 2, None, model._row_bytes("float32"))

    def plan(*_args, **_kwargs):
        return Plan(iter(tasks), original_plan.rows, False), window, stop

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native range invariant reached the fallback executor")

    monkeypatch.setattr(model, "plan_tasks", plan)
    monkeypatch.setattr(_pipeline, "run", forbidden)
    before = _native._test_team_calls()
    with pytest.raises(RuntimeError, match="native task team requires ordered, disjoint"):
        model.infer(source, workers=2)
    assert _native._test_team_calls() == before
    assert _native._test_team_resources() == 0


def test_checked_preparation_error_replays_tasks(monkeypatch):
    model = pipeline()
    sentences = texts(2000)
    source = StringSource(sentences)

    def checked_error(*_args):
        raise ValueError("checked output/input error")

    monkeypatch.setattr(model.backend, "string_team", checked_error)
    before = _native._test_team_calls()
    found = model.infer(source, workers=2)
    assert _native._test_team_calls() == before
    np.testing.assert_array_equal(
        found.view(np.uint32), expected(model.spec, sentences).view(np.uint32)
    )
