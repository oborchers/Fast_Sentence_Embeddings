"""Quiet waits retain strict load checks, owner isolation and finite budgets."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def waiting(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    module = importlib.import_module("compare_wait")
    now, sleeps = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.os, "getloadavg", lambda: (1, 1, 1), raising=False)

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(module.time, "sleep", sleep)
    return module, now, sleeps


def test_operation_starts_only_after_load_is_strictly_below_five(waiting, monkeypatch, tmp_path):
    module, now, sleeps = waiting
    loads = iter([(5, 1, 1), (4.99, 1, 1)])
    monkeypatch.setattr(module.os, "getloadavg", lambda: next(loads))
    seen = []
    module.ready_operation(tmp_path / "bench.lock", 120, lambda: seen.append(now[0]))
    assert seen == [60]
    assert sleeps == [60]
    assert not (tmp_path / "bench.lock").exists()


def test_budget_expiry_never_enters_an_operation(waiting, monkeypatch, tmp_path):
    module, _, sleeps = waiting
    monkeypatch.setattr(module.os, "getloadavg", lambda: (6, 1, 1))
    with pytest.raises(module.TimingBudgetExpired):
        module.ready_operation(tmp_path / "bench.lock", 20, lambda: pytest.fail("started"))
    assert sleeps == [20]


def test_wait_never_removes_another_owners_lock(waiting, tmp_path):
    module, _, _ = waiting
    lock = tmp_path / "bench.lock"
    lock.mkdir()
    (lock / "owner").write_text("other worker")
    with pytest.raises(module.TimingBudgetExpired):
        module.wait_for_quiet(lock, 10)
    assert (lock / "owner").read_text() == "other worker"


def test_lost_reservation_race_retries_outside_the_timed_operation(waiting, tmp_path):
    module, _, sleeps = waiting
    calls = []

    def operation():
        calls.append(1)
        if len(calls) == 1:
            raise module.TimingDeferred("load rose before reservation")
        return {"timed_seconds": 1.0}

    assert module.ready_operation(tmp_path / "bench.lock", 120, operation) == {"timed_seconds": 1.0}
    assert len(calls) == 2
    assert sleeps == [60]


def test_child_failure_is_not_retried_as_machine_contention(waiting, tmp_path):
    module, _, sleeps = waiting

    def operation():
        raise RuntimeError("child failed")

    with pytest.raises(RuntimeError, match="child failed"):
        module.ready_operation(tmp_path / "bench.lock", 120, operation)
    assert sleeps == []


@pytest.mark.parametrize("deadline", [float("nan"), float("inf")])
def test_nonfinite_budget_cannot_start_work(waiting, tmp_path, deadline):
    module, _, _ = waiting
    with pytest.raises(ValueError, match="finite"):
        module.wait_for_quiet(tmp_path / "bench.lock", deadline)
