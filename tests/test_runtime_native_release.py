"""Scan errors retain their traceback without retaining input owners until cyclic GC."""

import threading
import traceback
import weakref

import pytest

from fse import _native
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
