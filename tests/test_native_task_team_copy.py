"""Native task copies account every actual range, including failed-task replay."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fse import Encoder, _native
from fse._backends._cpu import CpuBackend
from fse._runtime import _native_run, _pipeline
from fse._runtime._worker import WorkerLocal
from fse._text import StringSource, copied_strings
from support.native_team import faults, pipeline, tasks_for
from support.runtime import expected, small_tasks, texts, word_vectors
from support.waiting import until


@pytest.fixture(autouse=True)
def balanced(monkeypatch):
    monkeypatch.setattr(_pipeline, "INLINE_MAX_BYTES", 0)
    baseline = (_native._test_team_resources(), _native._test_join_resources())
    yield
    _native._test_team_copy_control(-1)
    _native._string_copy_check(None)
    gc.collect()
    assert (_native._test_team_resources(), _native._test_join_resources()) == baseline


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("workers", [2, 4, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_success_accounts_every_task_and_keeps_reference_bits(
    monkeypatch, container, workers, normalize
):
    small_tasks(monkeypatch)
    model = pipeline(normalize, fasttext=True)
    rows = ["", "w1\0w2", "w3 " * 500, "".join(map(chr, range(128)))] * 40
    source = copied_strings(container(rows))
    tasks = tasks_for(model, source, workers)
    calls = _native._test_team_calls()
    actual = model.infer(source, workers=workers)
    assert _native._test_team_calls() == calls + 1
    np.testing.assert_array_equal(
        actual.view(np.uint32), expected(model.spec, rows).view(np.uint32)
    )
    assert _native._test_string_copy_progress(source._copy) == (len(rows), len(tasks))
    data, offsets = _native._test_string_copy_result(source._copy)
    assert data == "".join(rows).encode("ascii")
    assert np.diff(offsets).tolist() == [len(row) for row in rows]


def test_public_encode_uses_team_without_join_or_python_executor(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("eligible collection must use native copying")

    encoder = Encoder.from_word_vectors(word_vectors())
    rows = texts(2000)
    monkeypatch.setattr(_native, "join_strings", forbidden)
    monkeypatch.setattr(_pipeline, "run", forbidden)
    calls = _native._test_team_calls()
    result = encoder.encode(rows, workers=4)
    assert _native._test_team_calls() == calls + 1
    assert len(result) == len(rows)


@pytest.mark.parametrize("kind", [1, 4, 10])
def test_failed_task_replay_reuses_copy_and_accounts_later_tasks(monkeypatch, kind):
    small_tasks(monkeypatch)
    model = pipeline(fasttext=True)
    rows = texts(200)
    source = copied_strings(rows)
    tasks = tasks_for(model, source, 2)
    assert len(tasks) > 2
    _native._test_team_copy_control(0)
    previous = _native._test_corrupt(kind)
    checks = []

    def before_replay(first, last):
        # This runs before each error-localization prefix, after native accounting.
        checks.append((first, last))
        assert _native._test_string_copy_progress(source._copy) == (len(rows), len(tasks))
        assert _native._test_string_copy_result(source._copy)[0] == "".join(rows).encode()

    _native._string_copy_check(before_replay)
    try:
        with faults(corrupt_task=0), ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.infer, source, workers=2)
            try:
                assert until(lambda: _native._test_team_copy_control()[1] == len(tasks))
            finally:
                _native._test_team_copy_control(-1)
            with pytest.raises(ValueError, match=r".") as caught:
                future.result(timeout=15)
        native_error = type(caught.value), str(caught.value)
        del caught
        assert checks
        assert _native._test_string_copy_progress(source._copy) == (len(rows), len(tasks))
        _native._string_copy_check(None)
        monkeypatch.setattr(_native_run, "eligible", lambda *_args: False)
        with pytest.raises(ValueError, match=r".") as old:
            model.infer(copied_strings(rows), workers=2)
        assert native_error == (type(old.value), str(old.value))
    finally:
        _native._test_corrupt(previous)


def test_no_memory_accounts_copies_without_replaying(monkeypatch):
    small_tasks(monkeypatch)
    source = copied_strings(texts(200))
    model = pipeline()
    tasks = tasks_for(model, source, 2)
    _native._test_team_copy_control(0)

    def forbidden(*_args):
        raise AssertionError("memory failure must not replay")

    _native._string_copy_check(forbidden)
    with faults(status_task=0, status=1), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.infer, source, workers=2)
        try:
            assert until(lambda: _native._test_team_copy_control()[1] == len(tasks))
        finally:
            _native._test_team_copy_control(-1)
        with pytest.raises(MemoryError, match=r"^could not grow the buffers of the Workspace$"):
            future.result(timeout=15)
    assert _native._test_string_copy_progress(source._copy) == (len(source), len(tasks))


def test_skipped_tasks_are_not_counted_as_copied(monkeypatch):
    small_tasks(monkeypatch)
    source = copied_strings(texts(200))
    model = pipeline()
    tasks = tasks_for(model, source, 4)
    # All helper starts fail, so task zero fails before any other task can copy.
    with faults(failed_starts=254, status_task=0, status=1), pytest.raises(MemoryError):
        model.infer(source, workers=4)
    assert len(tasks) > 1
    assert _native._test_string_copy_progress(source._copy) == (tasks[0].n_rows, 1)
    with pytest.raises(AssertionError, match="not copied every row"):
        _native._test_string_copy_result(source._copy)


def test_preparation_fallback_copies_once_and_releases_team(monkeypatch):
    small_tasks(monkeypatch)
    source = copied_strings(texts(200))
    model = pipeline()
    tasks = tasks_for(model, source, 4)
    teams = []
    factory = CpuBackend.string_team

    def capture(*args):
        team = factory(*args)
        teams.append(team)
        return team

    monkeypatch.setattr(CpuBackend, "string_team", capture)
    calls = _native._test_team_calls()
    with faults(allocation=7):
        result = model.infer(source, workers=4)
    assert _native._test_team_calls() == calls
    assert len(teams) == 1
    assert not teams[0].prepared
    allocated, freed, started, joined, balanced, _ = teams[0].balance
    assert allocated == freed
    assert started == joined == 0
    assert balanced
    assert _native._test_string_copy_progress(source._copy) == (len(source), len(tasks))
    assert len(result) == len(source)


def test_held_owner_and_team_have_separate_resource_balances(monkeypatch):
    small_tasks(monkeypatch)
    baseline = _native._test_join_resources()
    source = copied_strings(texts(200))
    owner = weakref.ref(source._copy)
    teams = []
    factory = CpuBackend.string_team

    def capture(*args):
        team = factory(*args)
        teams.append(team)
        return team

    monkeypatch.setattr(CpuBackend, "string_team", capture)
    pipeline().infer(source, workers=4)
    assert _native._test_team_resources() == 0
    assert _native._test_join_resources() == baseline + 1
    assert teams[0].balance[:2] == (3, 3)
    assert teams[0].balance[4]
    del source
    # A held TaskTeam also retains the source until that team is released.
    assert owner() is not None
    teams.clear()
    gc.collect()
    assert owner() is None
    assert _native._test_join_resources() == baseline


def test_claimed_task_copies_after_an_earlier_failure(monkeypatch):
    small_tasks(monkeypatch)
    model = pipeline()
    source = copied_strings(texts(200))
    tasks = tasks_for(model, source, 2)
    _native._test_team_copy_control(1, before=True)
    previous = _native._test_corrupt(1)
    calls = _native._test_team_calls()
    try:
        with faults(corrupt_task=0), ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.infer, source, workers=2)
            try:
                assert until(lambda: _native._test_team_copy_control()[2:] == (0, 1))
                # Task zero failed. Task one was claimed but has not copied yet.
                assert _native._test_team_copy_control()[1] == 1
            finally:
                _native._test_team_copy_control(-1)
            with pytest.raises(ValueError, match=r"."):
                future.result(timeout=15)
    finally:
        _native._test_corrupt(previous)
    assert _native._test_team_calls() == calls + 1
    assert _native._test_string_copy_progress(source._copy) == (
        tasks[0].n_rows + tasks[1].n_rows,
        2,
    )


@pytest.mark.parametrize("source_factory", [StringSource, copied_strings])
@pytest.mark.parametrize("size", [0, 1, 4])
def test_malformed_native_task_item_is_rejected(source_factory, size):
    model = pipeline()
    source = source_factory(["w1", "w2"])
    worker = model._task_work(source, True, WorkerLocal(), 0, None)
    with pytest.raises(RuntimeError, match="native text task item has invalid copy state"):
        worker((None,) * size)


def test_joined_task_rejects_copy_state():
    model = pipeline()
    source = StringSource(["w1", "w2"])
    worker = model._task_work(source, True, WorkerLocal(), 0, None)
    with pytest.raises(RuntimeError, match="native text task item has invalid copy state"):
        worker((None, None, (True, 1)))
