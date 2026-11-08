"""Cross-process contract of the shared development locks, using private paths only."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

from support.waiting import announced

pytestmark = pytest.mark.skipif(os.name != "posix", reason="development locks require Unix")
BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
CONTENDER = Path(__file__).parent / "support" / "lock_contender.py"


@pytest.fixture
def locks(tmp_path, monkeypatch):
    monkeypatch.setenv("FSE_BENCH_LOCK", str(tmp_path / "bench.lock"))
    monkeypatch.setenv("FSE_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    import _development_locks

    return _development_locks


@contextmanager
def contender(role="heavy", mode="normal"):
    child = subprocess.Popen(
        [sys.executable, "-u", str(CONTENDER), role, mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def event(child, name):
    return announced(child.stdout, lambda item: item["event"] == name)


def finish(child):
    child.stdin.write(b"x")
    child.stdin.flush()
    event(child, "lock_released")
    assert child.wait(timeout=10) == 0, child.stderr.read().decode()


@pytest.mark.parametrize("role", ["heavy", "timing"])
def test_fifo_across_four_processes(locks, role):
    with ExitStack() as stack:
        holder = stack.enter_context(contender(role))
        event(holder, "lock_acquired")
        children = []
        for position in range(1, 4):
            child = stack.enter_context(contender(role))
            waiting = event(child, "waiting_for_lock")
            assert (waiting["position"], waiting["waiting"]) == (position, position)
            assert waiting["holder"]["pid"] == holder.pid
            children.append(child)
        finish(holder)
        for child in children:
            owner = event(child, "lock_acquired")["owner"]
            assert owner["pid"] == child.pid
            assert set(owner) == {"pid", "worker", "purpose", "token"}
            finish(child)
    root = locks.lock_paths()[role == "heavy"]
    assert [p.name for p in root.with_name(root.name + ".queue").iterdir()] == [".guard"]


@pytest.mark.parametrize("role", ["heavy", "timing"])
@pytest.mark.parametrize("stop", ["signal", "kill", "budget"])
def test_canceled_waiter_does_not_block(locks, role, stop):
    with contender(role) as holder:
        event(holder, "lock_acquired")
        with contender(role, "budget" if stop == "budget" else "normal") as first:
            event(first, "waiting_for_lock")
            if stop != "budget":
                first.send_signal(signal.SIGKILL if stop == "kill" else signal.SIGTERM)
            assert first.wait(timeout=10) == (-signal.SIGKILL if stop == "kill" else 0)
            with contender(role) as next_child:
                assert event(next_child, "waiting_for_lock")["position"] == 1
                finish(holder)
                event(next_child, "lock_acquired")
                finish(next_child)
    root = locks.lock_paths()[role == "heavy"]
    assert not root.exists()
    assert [p.name for p in root.with_name(root.name + ".queue").iterdir()] == [".guard"]


@pytest.mark.parametrize(("role", "peer"), [("heavy", "timing"), ("timing", "heavy")])
def test_peer_exclusion(locks, role, peer):
    with contender(role) as holder:
        event(holder, "lock_acquired")
        with contender(peer) as waiting:
            message = event(waiting, "waiting_for_lock")
            assert message["peer_holder"]["pid"] == holder.pid
            assert not locks.lock_paths()[peer == "heavy"].exists()
            finish(holder)
            event(waiting, "lock_acquired")
            finish(waiting)


def test_old_holder_is_not_taken_over(locks):
    path = locks.lock_paths()[1]
    path.mkdir()
    (path / "owner").write_text('{"pid": -1}')
    with contender() as child:
        assert event(child, "waiting_for_lock")["holder"] == {"pid": -1}
        child.terminate()
        event(child, "stopped")
        assert child.wait(timeout=10) == 0
    assert json.loads((path / "owner").read_text()) == {"pid": -1}


@pytest.mark.parametrize("poll", [0, -1, float("inf"), float("nan")])
def test_poll_validation(locks, poll):
    with (
        pytest.raises(ValueError, match="finite and positive"),
        locks.coordinated("heavy", "test", "test", lambda _: None, poll=poll),
    ):
        pytest.fail("invalid poll admitted")


def test_busy_includes_broken_symlinks(locks, tmp_path):
    path = tmp_path / "link"
    path.symlink_to(tmp_path / "missing")
    assert locks.busy(path)


@pytest.mark.parametrize("configuration", ["same", "nested", "queue", "nested_queue"])
def test_reject_overlapping_paths(locks, monkeypatch, tmp_path, configuration):
    bench = tmp_path / "bench.lock"
    heavy = {
        "same": bench,
        "nested": bench / "heavy",
        "queue": tmp_path / "bench.lock.queue",
        "nested_queue": tmp_path / "bench.lock.queue" / "heavy",
    }[configuration]
    monkeypatch.setenv("FSE_HEAVY_LOCK", str(heavy))
    with pytest.raises(ValueError, match="paths"):
        locks.lock_paths()
    assert not bench.exists()


@pytest.mark.parametrize("failure", ["notify", "admitted", "payload", "ticket"])
def test_acquisition_exceptions_release_lock_and_ticket(locks, monkeypatch, failure):
    import _development_queue

    calls = 0
    real = _development_queue.Waiter.close

    def close(waiter):
        real(waiter)
        if failure == "ticket":
            raise RuntimeError("ticket failure")

    def admitted():
        nonlocal calls
        calls += 1
        if failure == "admitted" and calls == 2:
            raise RuntimeError("admission failure")
        return True

    def notify(message):
        if failure == "notify" and message["event"] == "lock_acquired":
            raise RuntimeError("notification failure")

    monkeypatch.setattr(_development_queue.Waiter, "close", close)
    with (
        pytest.raises(RuntimeError, match="failure"),
        locks.coordinated("heavy", "test", "test", notify, admitted=admitted, poll=0.01),
    ):
        raise RuntimeError("payload failure")
    assert not locks.lock_paths()[1].exists()
    assert [p.name for p in (locks.lock_paths()[1].with_name("heavy.lock.queue")).iterdir()] == [
        ".guard"
    ]


def test_second_admission_retains_head_ticket(locks):
    calls, messages = 0, []

    def admitted():
        nonlocal calls
        calls += 1
        return calls != 2

    def notify(message):
        messages.append(message)
        if message["event"] == "waiting_for_lock":
            assert message["position"] == 1
            assert not locks.lock_paths()[1].exists()

    with locks.coordinated("heavy", "test", "test", notify, admitted=admitted, poll=0.001):
        assert calls == 4
    assert [m["event"] for m in messages] == ["waiting_for_lock", "lock_acquired", "lock_released"]
