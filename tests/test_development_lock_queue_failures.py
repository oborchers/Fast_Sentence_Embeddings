"""Failure and lifecycle probes for private development-lock queues."""

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from support.waiting import announced

pytestmark = pytest.mark.skipif(os.name != "posix", reason="development locks require Unix")


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setenv("FSE_BENCH_LOCK", str(tmp_path / "bench.lock"))
    monkeypatch.setenv("FSE_HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import _development_queue

    return _development_queue


def waiter(queue, tmp_path):
    return queue.Waiter(tmp_path / "heavy.lock", {"token": uuid.uuid4().hex})


@pytest.mark.parametrize("entry", ["unknown", "symlink", "directory", "guard_symlink"])
def test_corrupt_queue_fails_closed(queue, tmp_path, entry):
    current = waiter(queue, tmp_path)
    current.root.mkdir()
    ticket = current.root / ("000000000001-" + "a" * 32)
    if entry == "unknown":
        (current.root / "foreign").touch()
    elif entry == "symlink":
        ticket.symlink_to(tmp_path / "absent")
    elif entry == "directory":
        ticket.mkdir()
    else:
        (current.root / ".guard").symlink_to(tmp_path / "absent")
    try:
        with pytest.raises((RuntimeError, OSError)):
            current.position()
    finally:
        current.close()
    assert (
        current.root
        / (
            "foreign"
            if entry == "unknown"
            else ".guard"
            if entry == "guard_symlink"
            else ticket.name
        )
    ).lstat()


def test_symlink_queue_rejected(queue, tmp_path):
    current = waiter(queue, tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    current.root.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        current.position()
    assert not list(target.iterdir())


@pytest.mark.parametrize("change", ["missing", "replaced"])
def test_own_ticket_identity_checked(queue, tmp_path, change):
    current = waiter(queue, tmp_path)
    assert current.position() == (1, 1)
    # Preserve the old inode so replacement cannot reuse it.
    current.ticket.rename(tmp_path / "saved-ticket")
    if change == "replaced":
        current.ticket.touch()
    with pytest.raises(RuntimeError, match=r"disappeared|changed"):
        current.position()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        current.close()
    if change == "replaced":
        assert current.ticket.exists()


@pytest.mark.parametrize("stage", ["before_flock", "during_json"])
def test_crash_during_registration_is_reaped(queue, tmp_path, stage):
    # The crashing child announces from inside registration while holding the guard.
    script = """
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import _development_queue as q
w = q.Waiter(Path(sys.argv[2]), {"token": "a" * 32})
real = q._try_lock
calls = 0
def lock(fd):
    global calls
    calls += 1
    if calls == 2 and sys.argv[3] == "before_flock":
        print(json.dumps({"event": "crash"}), flush=True)
        os._exit(9)
    return real(fd)
def dump(*args):
    print(json.dumps({"event": "crash"}), flush=True)
    os._exit(9)
q._try_lock = lock
q.json.dump = dump
w.position()
"""
    with subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            script,
            str(Path(queue.__file__).parent),
            str(tmp_path / "heavy.lock"),
            stage,
        ],
        stdout=subprocess.PIPE,
    ) as child:
        try:
            announced(child.stdout, lambda item: item["event"] == "crash")
            assert child.wait(timeout=10) == 9
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
    current = waiter(queue, tmp_path)
    try:
        assert current.position() == (1, 1)
        assert len(list(current.root.iterdir())) == 2
    finally:
        current.close()


def test_busy_guard_keeps_waiter_abortable(queue, tmp_path):
    import fcntl

    current = waiter(queue, tmp_path)
    current.root.mkdir()
    with (current.root / ".guard").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = Path(__file__).parent / "support" / "lock_contender.py"
        child = subprocess.Popen(
            [sys.executable, "-u", str(script), "heavy", "normal"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            message = announced(child.stdout, lambda item: item["event"] == "waiting_for_lock")
            assert message["position"] is None
            child.terminate()
            announced(child.stdout, lambda item: item["event"] == "stopped")
            assert child.wait(timeout=10) == 0
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
    assert list(current.root.iterdir()) == [current.root / ".guard"]


def test_ticket_write_failure_closes_descriptor(queue, tmp_path, monkeypatch):
    current = waiter(queue, tmp_path)

    def fail(*_args):
        raise OSError("write failed")

    monkeypatch.setattr(queue.json, "dump", fail)
    try:
        with pytest.raises(OSError, match="write failed"):
            current.position()
    finally:
        current.close()
    assert current.fd is None
    assert list(current.root.iterdir()) == [current.root / ".guard"]


def test_registered_waiter_can_leave_with_guard_held(queue, tmp_path):
    import fcntl

    current = waiter(queue, tmp_path)
    assert current.position() == (1, 1)
    with (current.root / ".guard").open("r+") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert current.position() is None
        current.close()
    assert list(current.root.iterdir()) == [current.root / ".guard"]


def test_concurrent_cancellation_while_reaping(queue, tmp_path, monkeypatch):
    current = waiter(queue, tmp_path)
    current.root.mkdir()
    abandoned = current.root / ("000000000001-" + "a" * 32)
    abandoned.touch()
    real = queue._unlink

    def disappearing(path, fd):
        if path == abandoned:
            path.unlink()
        real(path, fd)

    monkeypatch.setattr(queue, "_unlink", disappearing)
    try:
        assert current.position() == (1, 1)
    finally:
        current.close()
