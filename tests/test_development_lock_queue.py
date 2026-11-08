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
    return announced(
        child.stdout,
        lambda item: (
            item["event"] == name
            and (name != "waiting_for_lock" or isinstance(item["position"], int))
        ),
    )


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


@pytest.mark.usefixtures("locks")
def test_paused_head_is_not_reaped():
    with contender() as holder:
        event(holder, "lock_acquired")
        with contender() as head:
            event(head, "waiting_for_lock")
            head.send_signal(signal.SIGSTOP)
            try:
                with contender() as later:
                    assert event(later, "waiting_for_lock")["position"] == 2
                    later.terminate()
                    event(later, "stopped")
                    assert later.wait(timeout=10) == 0
            finally:
                head.send_signal(signal.SIGCONT)
            head.terminate()
            event(head, "stopped")
            assert head.wait(timeout=10) == 0
        finish(holder)


def test_peer_appearing_after_claim_releases_own_lock(locks, monkeypatch):
    bench, heavy = locks.lock_paths()
    real = locks._claim
    first = True
    messages = []

    def claim(path, owner):
        nonlocal first
        result = real(path, owner)
        if first:
            bench.mkdir()
            first = False
        return result

    def notify(message):
        messages.append(message)
        if message["event"] == "waiting_for_lock":
            assert not heavy.exists()
            assert message["position"] == 1
            bench.rmdir()

    monkeypatch.setattr(locks, "_claim", claim)
    with locks.coordinated("heavy", "test", "test", notify, poll=0.001):
        assert heavy.exists()
        assert not bench.exists()
    assert [m["event"] for m in messages] == ["waiting_for_lock", "lock_acquired", "lock_released"]


@pytest.mark.usefixtures("locks")
def test_legacy_waiter_preserves_exclusion():
    # Use the pre-queue mkdir protocol in another process, sharing only private paths.
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from _development_locks import _claim, _release, lock_paths
path = lock_paths()[1]
owner = {"worker": "legacy", "pid": -1, "purpose": "test", "token": "legacy"}
assert _claim(path, owner)
print(json.dumps({"event": "legacy_acquired"}), flush=True)
sys.stdin.buffer.read(1)
_release(path, owner)
"""
    with subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(BENCHMARKS)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as old:
        try:
            event(old, "legacy_acquired")
            with contender() as new:
                assert event(new, "waiting_for_lock")["holder"]["worker"] == "legacy"
                old.stdin.write(b"x")
                old.stdin.flush()
                assert old.wait(timeout=10) == 0
                event(new, "lock_acquired")
                finish(new)
        finally:
            if old.poll() is None:
                old.kill()
            old.communicate(timeout=10)


@pytest.mark.parametrize("stage", ["registration", "notification"])
@pytest.mark.parametrize("stop", ["signal", "budget"])
def test_stop_before_payload(locks, monkeypatch, stage, stop):
    from types import SimpleNamespace

    import _development_budget
    import _development_queue
    from _harness.budget import WallBudgetExpiredError

    state = SimpleNamespace(wall_deadline=None, stopping=False)
    real = _development_queue.Waiter.position

    def halt():
        if stop == "budget":
            state.wall_deadline = 0
        else:
            state.stopping = True

    def position(waiter):
        result = real(waiter)
        if stage == "registration":
            halt()
        return result

    def notify(message):
        if stage == "notification" and message["event"] == "lock_acquired":
            halt()

    monkeypatch.setattr(_development_queue.Waiter, "position", position)
    expected = WallBudgetExpiredError if stop == "budget" else InterruptedError
    with (
        pytest.raises(expected),
        locks.coordinated(
            "heavy", "test", "test", notify, lambda: _development_budget.stopped(state)
        ),
    ):
        pytest.fail("stopped process started payload")
    assert not locks.lock_paths()[1].exists()
    assert [p.name for p in locks.lock_paths()[1].with_name("heavy.lock.queue").iterdir()] == [
        ".guard"
    ]


@pytest.mark.parametrize("kind", ["fifo", "symlink", "oversized", "partial"])
def test_holder_snapshot_is_nonblocking_and_bounded(locks, tmp_path, kind):
    holder = tmp_path / "holder"
    holder.mkdir()
    owner = holder / "owner"
    if kind == "fifo":
        os.mkfifo(owner)
    elif kind == "symlink":
        other = tmp_path / "target"
        other.write_text('{"private": true}')
        owner.symlink_to(other)
    else:
        owner.write_text("x" * 65537 if kind == "oversized" else '{"pid":')
    assert locks._holder(holder) is None


@pytest.mark.usefixtures("locks")
@pytest.mark.parametrize("roles", [("timing", "heavy"), ("legacy", "heavy")])
def test_barrier_started_contenders_do_not_overlap(roles):
    import selectors
    import time

    script = CONTENDER.with_name("lock_racer.py")
    children = []
    try:
        for role in roles:
            child = subprocess.Popen(
                [sys.executable, "-u", str(script), role],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            children.append(child)
            event(child, "ready")
        for child in children:
            child.stdin.write(b"x")
            child.stdin.flush()
        with selectors.DefaultSelector() as selector:
            for child in children:
                selector.register(child.stdout, selectors.EVENT_READ, child)
            winner = None
            deadline = time.monotonic() + 10
            while winner is None:
                remaining = deadline - time.monotonic()
                assert remaining > 0, "neither contender entered payload before deadline"
                ready = selector.select(remaining)
                assert ready, "neither contender entered payload before deadline"
                for key, _ in ready:
                    child = key.data
                    message = announced(child.stdout, lambda _: True)
                    if message["event"] == "payload":
                        winner = child
                        break
        loser = next(child for child in children if child is not winner)
        event(loser, "waiting_for_lock")
        finish(winner)
        event(loser, "payload")
        finish(loser)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
