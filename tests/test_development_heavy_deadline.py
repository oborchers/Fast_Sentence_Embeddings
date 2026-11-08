"""The heavy lock outlives leaders and descendants, including deadline escalation."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from support.waiting import until

pytestmark = pytest.mark.skipif(os.name != "posix", reason="heavy process groups require Unix")
HELPER = Path(__file__).parent / "support" / "heavy_descendant.py"


@pytest.fixture
def heavy(tmp_path, monkeypatch):
    monkeypatch.setenv("FSE_BENCH_LOCK", str(tmp_path / "bench.lock"))
    monkeypatch.setenv("FSE_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import _development_heavy

    return _development_heavy


def _run(heavy, root, mode, deadline=None):
    state = SimpleNamespace(number=None, child=None, received_at=None)
    result, errors = [], []

    def worker():
        try:
            with heavy.held(state, "test-heavy", "descendant lifetime", lambda _: None):
                result.append(
                    heavy.execute(
                        [sys.executable, str(HELPER), str(root), "leader", mode],
                        state,
                        lambda _: None,
                        deadline=deadline,
                    )
                )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    return state, thread, result, errors


def test_lock_remains_until_descendant_finishes(heavy, tmp_path):
    state, thread, result, errors = _run(heavy, tmp_path, "normal")
    try:
        assert until(lambda: (tmp_path / "descendant.pid").exists())
        assert until(lambda: state.child.poll() is not None)
        assert (tmp_path / "heavy.lock" / "owner").is_file()
        assert thread.is_alive()
    finally:
        (tmp_path / "release").touch()
        thread.join(10)
    assert not thread.is_alive()
    assert errors == []
    assert result[0][0] == 0
    assert result[0][1]["escalated"] is None
    assert not (tmp_path / "heavy.lock").exists()


@pytest.mark.parametrize(("mode", "escalation"), [("normal", "term"), ("ignore", "kill")])
def test_deadline_drains_group_before_lock_release(heavy, tmp_path, monkeypatch, mode, escalation):
    bounded, drain = heavy._bounded, heavy._drain

    def after_ready(child, state, notify, _deadline):
        assert until(lambda: (tmp_path / "descendant.pid").exists())
        assert until(lambda: child.poll() is not None)
        assert (tmp_path / "heavy.lock" / "owner").is_file()
        return bounded(child, state, notify, time.monotonic())

    monkeypatch.setattr(heavy, "_bounded", after_ready)
    monkeypatch.setattr(heavy, "_drain", lambda child, notify: drain(child, notify, grace=0.2))
    _, thread, result, errors = _run(heavy, tmp_path, mode, deadline=time.monotonic() + 60)
    try:
        thread.join(10)
        assert not thread.is_alive()
        assert errors == []
        assert result[0][1]["deadline"] is True
        assert result[0][1]["escalated"] == escalation
        assert not (tmp_path / "heavy.lock").exists()
        descendant = int((tmp_path / "descendant.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(descendant, 0)
    finally:
        (tmp_path / "release").touch()
        thread.join(10)


@pytest.mark.parametrize("status", [0, 7])
def test_normal_wrapper_keeps_integer_exit_status(heavy, status):
    value = heavy.run(
        [sys.executable, "-c", f"raise SystemExit({status})"], "test-heavy", lambda _: None
    )
    assert type(value) is int
    assert value == status
