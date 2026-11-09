"""The heavy lock outlives leaders and descendants, including deadline escalation."""

from __future__ import annotations

import os
import signal
import subprocess
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
    state = heavy._Signals()
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


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin zombie-only killpg behavior")
def test_zombie_only_group_is_proven_ended_under_lock(heavy, tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], start_new_session=True)
    state = SimpleNamespace(number=None)
    try:
        with heavy.held(state, "test-heavy", "Darwin zombie group", lambda _: None):

            def zombie():
                output = subprocess.check_output(
                    ["/bin/ps", "-p", str(child.pid), "-o", "stat="], text=True
                )
                return output.strip().startswith("Z")

            assert until(zombie)
            with pytest.raises(PermissionError):
                os.killpg(child.pid, 0)
            assert heavy._group_alive(child.pid) is False
            heavy._send(child.pid, signal.SIGTERM)
            heavy._send(child.pid, signal.SIGKILL)
            assert (tmp_path / "heavy.lock" / "owner").is_file()
    finally:
        assert child.wait(timeout=5) == 7
    assert not (tmp_path / "heavy.lock").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin permission fallback")
def test_live_member_does_not_turn_permission_error_into_completion(heavy, tmp_path, monkeypatch):
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    real = heavy.os.killpg

    def denied(group, signum):
        if group == child.pid and signum == 0:
            raise PermissionError("probe denied")
        return real(group, signum)

    try:
        with heavy.held(SimpleNamespace(number=None), "test-heavy", "live member", lambda _: None):
            monkeypatch.setattr(heavy.os, "killpg", denied)
            with pytest.raises(PermissionError, match="probe denied"):
                heavy._group_alive(child.pid)
            assert (tmp_path / "heavy.lock" / "owner").is_file()
    finally:
        child.communicate(b"x", timeout=10)
    assert child.returncode == 0
    assert not (tmp_path / "heavy.lock").exists()


def test_group_query_failure_keeps_lock_through_cleanup(heavy, tmp_path, monkeypatch):
    real_alive, real_send, real_drain = heavy._group_alive, heavy._send, heavy._drain
    failures = []

    def denied_twice(group):
        if len(failures) < 2:
            assert until(lambda: (tmp_path / "descendant.pid").exists())
            failures.append(group)
            raise PermissionError("probe denied")
        return real_alive(group)

    def send(group, signum):
        assert (tmp_path / "heavy.lock" / "owner").is_file()
        return real_send(group, signum)

    monkeypatch.setattr(heavy, "_group_alive", denied_twice)
    monkeypatch.setattr(heavy, "_send", send)
    monkeypatch.setattr(
        heavy,
        "_drain",
        lambda child, notify, **kwargs: real_drain(child, notify, grace=0.2, **kwargs),
    )
    _, thread, result, errors = _run(heavy, tmp_path, "ignore", deadline=time.monotonic() + 60)
    try:
        thread.join(10)
        assert not thread.is_alive()
        assert result == []
        assert len(errors) == 1
        assert isinstance(errors[0], PermissionError)
        assert len(failures) == 2
        assert not (tmp_path / "heavy.lock").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "descendant.pid").read_text()), 0)
    finally:
        (tmp_path / "release").touch()
        thread.join(10)


@pytest.mark.parametrize("iteration", range(10))
def test_single_process_deadline_records_clean_shutdown(heavy, tmp_path, iteration):
    state = SimpleNamespace(number=None, child=None, received_at=None)
    with heavy.held(state, f"test-heavy-{iteration}", "single process deadline", lambda _: None):
        status, metadata = heavy.execute(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            state,
            lambda _: None,
            deadline=time.monotonic() + 0.05,
        )
        assert status == -signal.SIGTERM
        assert metadata["deadline"] is True
        assert metadata["escalated"] == "term"
        assert (tmp_path / "heavy.lock" / "owner").is_file()
    assert not (tmp_path / "heavy.lock").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin complete process snapshot")
def test_empty_group_requires_a_complete_process_snapshot(heavy, monkeypatch):
    monkeypatch.setattr(heavy.subprocess, "check_output", lambda *_args, **_kwargs: "")
    with pytest.raises(RuntimeError, match="snapshot omitted its controller"):
        heavy._zombie_group(999999)
    monkeypatch.setattr(
        heavy.subprocess,
        "check_output",
        lambda *_args, **_kwargs: f"{os.getpid()} {os.getpgrp()} S\n",
    )
    assert heavy._zombie_group(999999) is True


def test_repeated_signal_failure_keeps_lock_until_group_ends(heavy, tmp_path, monkeypatch):
    real_send, real_drain = heavy._send, heavy._drain
    draining = threading.Event()
    injections, states = [], []
    bounded = heavy._bounded

    def denied(group, signum):
        if signum == signal.SIGINT:
            assert (tmp_path / "heavy.lock" / "owner").is_file()
            raise PermissionError("repeated signal denied")
        if draining.is_set() and not injections:
            # The drain always sends its first signal, so both repeated signals arrive
            # while it drains without depending on a poll, a sleep or the grace period.
            for _ in range(2):
                injections.append(True)
                states[0].receive(signal.SIGINT, None)
                assert (tmp_path / "heavy.lock" / "owner").is_file()
        return real_send(group, signum)

    def ready(child, state, notify, _deadline):
        assert until(lambda: states and (tmp_path / "descendant.pid").exists())
        return bounded(child, state, notify, time.monotonic())

    def drain(child, notify, **kwargs):
        draining.set()
        try:
            return real_drain(child, notify, grace=0.2, **kwargs)
        finally:
            draining.clear()

    monkeypatch.setattr(heavy, "_send", denied)
    monkeypatch.setattr(heavy, "_bounded", ready)
    monkeypatch.setattr(heavy, "_drain", drain)
    state, thread, result, errors = _run(heavy, tmp_path, "ignore", time.monotonic() + 0.1)
    states.append(state)
    try:
        thread.join(10)
        assert not thread.is_alive()
        assert len(injections) == 2
        assert result == []
        assert len(errors) == 1
        assert str(errors[0]) == "repeated signal denied"
        assert not (tmp_path / "heavy.lock").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "descendant.pid").read_text()), 0)
    finally:
        (tmp_path / "release").touch()
        thread.join(10)


def test_normal_wrapper_preserves_separate_error_stream(heavy, monkeypatch):
    real = heavy.subprocess.Popen
    streams = []

    def launch(*args, **kwargs):
        streams.append(kwargs["stderr"])
        return real(*args, **kwargs)

    monkeypatch.setattr(heavy.subprocess, "Popen", launch)
    assert heavy.run([sys.executable, "-c", "pass"], "test-heavy", lambda _: None) == 0
    assert streams[0] is None
