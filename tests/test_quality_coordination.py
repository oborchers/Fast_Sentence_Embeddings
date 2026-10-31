"""A quality job yields between tasks and never removes another worker's lock."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def coordination(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("quality_coordination")


def test_task_releases_lock_on_error(coordination, tmp_path):
    with pytest.raises(ValueError, match="failed"), coordination.task_reservation(tmp_path):
        raise ValueError("failed")
    assert not (tmp_path / "heavy.lock").exists()


def test_paused_task_waits_before_lock(coordination, tmp_path, monkeypatch):
    marker = tmp_path / "timing-window.marker"
    marker.touch()
    waits = []

    def sleep(seconds):
        assert not (tmp_path / "heavy.lock").exists()
        waits.append(seconds)
        marker.unlink()

    monkeypatch.setattr(coordination.time, "sleep", sleep)
    with coordination.task_reservation(tmp_path):
        assert waits == [60]


def test_other_lock_preserved_while_waiting(coordination, tmp_path, monkeypatch):
    lock = tmp_path / "heavy.lock"
    lock.mkdir()
    (lock / "owner").write_text("other")

    def sleep(_seconds):
        assert (lock / "owner").read_text() == "other"
        raise TimeoutError("still busy")

    monkeypatch.setattr(coordination.time, "sleep", sleep)
    with pytest.raises(TimeoutError, match="still busy"), coordination.task_reservation(tmp_path):
        pytest.fail("entered occupied lock")
    assert (lock / "owner").read_text() == "other"
