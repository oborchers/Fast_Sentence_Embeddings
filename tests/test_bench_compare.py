"""Shared-machine timing must fail closed and leave other owners alone."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    module = importlib.import_module("bench_compare")
    monkeypatch.setattr(module.os, "getloadavg", lambda: (1, 1, 1))
    return module


def test_reservation_preserves_existing_owner(bench, tmp_path):
    lock = tmp_path / "bench.lock"
    lock.mkdir()
    (lock / "owner").write_text("other worker")
    with pytest.raises(FileExistsError), bench.reservation(lock):
        pytest.fail("entered another worker's reservation")
    assert (lock / "owner").read_text() == "other worker"


def test_reservation_releases_after_failure(bench, tmp_path):
    lock = tmp_path / "bench.lock"
    with pytest.raises(ValueError), bench.reservation(lock):
        raise ValueError("child failed")
    assert not lock.exists()


@pytest.mark.parametrize("reason", ["load", "pause"])
def test_reservation_defers_before_timing(bench, tmp_path, monkeypatch, reason):
    lock = tmp_path / "bench.lock"
    if reason == "load":
        monkeypatch.setattr(bench.os, "getloadavg", lambda: (5, 1, 1))
    else:
        (tmp_path / "pause-local.marker").touch()
    with pytest.raises(RuntimeError), bench.reservation(lock):
        pytest.fail("timed during pause or high load")
    assert not lock.exists()


def test_reservation_rechecks_load_after_acquisition(bench, tmp_path, monkeypatch):
    loads = iter([(1, 1, 1), (5, 1, 1)])
    monkeypatch.setattr(bench.os, "getloadavg", lambda: next(loads))
    lock = tmp_path / "bench.lock"
    with pytest.raises(RuntimeError), bench.reservation(lock):
        pytest.fail("load rose while reserving")
    assert not lock.exists()
