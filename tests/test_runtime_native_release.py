"""Scan errors retain their traceback without retaining input owners until cyclic GC."""

import gc
import threading
import traceback
import weakref
from contextlib import contextmanager

import pytest

from fse import _native
from fse._backends._cpu import CpuBackend
from fse._runtime import Pipeline
from fse._runtime._tasks import Plan, Task
from fse._text import Chunk, StringSource, copied_strings
from support.native_team import faults, pipeline


def _scan_error_plan(_model, source, *_args, **_kwargs):
    def scan():
        yield Task(source, Chunk(0, len(source), 0, len(source)))
        raise ValueError("saved scan error")

    return Plan(scan(), len(source), False), 8, threading.Event()


@pytest.mark.parametrize("factory", [StringSource, copied_strings])
@pytest.mark.parametrize("preparation_failure", [False, True])
@pytest.mark.parametrize("workers", [4, 100])
def test_scan_error_releases_owners_without_gc(monkeypatch, factory, preparation_failure, workers):
    model = pipeline()
    monkeypatch.setattr(Pipeline, "plan_tasks", _scan_error_plan)

    def failing_call():
        source = factory(["w0 " * 100] * 512)
        reference = weakref.ref(source)
        owner = weakref.ref(source._copy) if factory is copied_strings else None
        try:
            model.infer(source, workers=workers)
        except ValueError as error:
            observed = (
                type(error),
                str(error),
                tuple(frame.name for frame in traceback.extract_tb(error.__traceback__)),
            )
        else:
            raise AssertionError("scan error was hidden")
        return reference, owner, observed

    with _without_cyclic_gc():
        before = _native._test_team_calls()
        with faults(allocation=int(preparation_failure)):
            source, owner, observed = failing_call()
        assert observed[:2] == (ValueError, "saved scan error")
        assert observed[2][-1] == "scan"
        assert "_buffer" in observed[2]
        assert ("_reraise" in observed[2]) == preparation_failure
        assert _native._test_team_calls() == before + (not preparation_failure)
        assert source() is None
        assert owner is None or owner() is None


@contextmanager
def _without_cyclic_gc():
    gc.collect()  # Clear earlier tests only. The call and lifetime assertions cannot collect.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class _InterruptedTeam:
    def __init__(self, team):
        self.team = team

    def __getattr__(self, name):
        return getattr(self.team, name)

    def run(self):
        self.team.run()  # Exercise the post-native KeyboardInterrupt cleanup boundary.
        raise KeyboardInterrupt


@pytest.mark.parametrize("factory", [StringSource, copied_strings])
@pytest.mark.parametrize("failure", ["row", "memory", "interrupt"])
def test_native_failures_release_owners_without_gc(monkeypatch, factory, failure):
    model = pipeline()
    if failure == "interrupt":
        original = CpuBackend.string_team

        def interrupted(*args):
            return _InterruptedTeam(original(*args))

        monkeypatch.setattr(CpuBackend, "string_team", interrupted)

    def failing_call():
        source = factory(["w0 " * 100] * 512)
        reference = weakref.ref(source)
        owner = weakref.ref(source._copy) if factory is copied_strings else None
        try:
            model.infer(source, workers=4)
        except BaseException as error:
            observed = (
                type(error),
                tuple(frame.name for frame in traceback.extract_tb(error.__traceback__)),
            )
        else:
            raise AssertionError("native failure was hidden")
        return reference, owner, observed

    corrupt = _native._test_corrupt(1 if failure == "row" else 0)
    try:
        with _without_cyclic_gc(), faults(status_task=0 if failure == "memory" else -1, status=1):
            before = _native._test_team_calls()
            source, owner, observed = failing_call()
            assert (
                observed[0]
                is {"row": ValueError, "memory": MemoryError, "interrupt": KeyboardInterrupt}[
                    failure
                ]
            )
            assert ("earliest" in observed[1]) == (failure == "row")
            assert _native._test_team_calls() == before + 1
            assert _native._test_team_resources() == 0
            assert source() is None
            assert owner is None or owner() is None
    finally:
        _native._test_corrupt(corrupt)
