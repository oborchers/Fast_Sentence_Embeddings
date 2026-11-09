"""Signal-forwarding Unix heavy jobs that own their lock until the group finishes."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

from _development_locks import coordinated
from _development_system import LABEL, require_unix
from _harness.budget import check


def _send(group, signum):
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        return
    except PermissionError:
        if sys.platform == "darwin" and not _group_alive(group):
            return
        raise


def _wait_group(group):
    while True:
        _reap(group)
        if not _group_alive(group):
            return
        time.sleep(0.05)


def _reap(group):
    # A wrapper running as Unix PID 1 may adopt descendants after their leader exits.
    while True:
        try:
            child, _ = os.waitpid(-group, os.WNOHANG)
        except ChildProcessError:
            return
        if child == 0:
            return


class _Signals:
    def __init__(self):
        self.number = None
        self.child = None
        self.received_at = None
        self.errors = []
        self.previous_mask = None

    def receive(self, signum, _frame):
        self.received_at = self.received_at or time.monotonic()
        self.number = self.number or signum
        if self.child is not None:
            _cleanup_call(lambda: _send(self.child.pid, signum), self.errors)

    def finish(self):
        # End signal acceptance before committing the terminal records. Restore the
        # original handlers before delivering signals arriving after this boundary.
        self.previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
        )
        for signum in signal.sigpending() & {signal.SIGINT, signal.SIGTERM}:
            self.receive(signum, None)


@contextmanager
def signals():
    state = _Signals()
    previous = {n: signal.signal(n, state.receive) for n in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield state
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if state.previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, state.previous_mask)


def _priority(command, notify):
    nice = shutil.which("nice")
    current = os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else None
    increment = max(0, 10 - current) if current is not None else 10
    notify(
        {
            **LABEL,
            "event": "heavy_priority",
            "parent_nice": current,
            "nice_increment": increment if nice else None,
        }
    )
    return [nice, "-n", str(increment), *command] if nice else command


@contextmanager
def held(state, worker, purpose, notify, deadline=None):
    def stopped():
        check(deadline)
        return state.number is not None

    with coordinated("heavy", worker, purpose, notify, stopped):
        yield


def _zombie_group(group):
    # Darwin filters zombie members even for real signals, not only signal zero.
    output = subprocess.check_output(["/bin/ps", "-axo", "pid=,pgid=,stat="], text=True)
    rows = [
        (int(pid), int(pgid), state)
        for pid, pgid, state in (line.split() for line in output.splitlines())
    ]
    if not any(pid == os.getpid() for pid, _, _ in rows):
        raise RuntimeError("Darwin process snapshot omitted its controller")
    # A complete snapshot can also prove that the last member was already reaped.
    return all(state.startswith("Z") for _, pgid, state in rows if pgid == group)


def _group_alive(group):
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        if sys.platform == "darwin" and _zombie_group(group):
            return False
        raise
    return True


def _alive(child):
    if child.poll() is not None:
        _reap(child.pid)
    return _group_alive(child.pid)


def _cleanup_call(callback, errors, default=None):
    try:
        return callback()
    except Exception as error:
        if not errors:
            errors.append(error)
        return default


def _ended_within(child, deadline, errors):
    while _cleanup_call(lambda: _alive(child), errors, True):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _drained(errors, escalated):
    if errors:
        raise errors[0]
    return escalated


def _drain(child, notify, grace=10.0, *, kill=True):
    errors = []
    escalated = "term"
    for signum in (signal.SIGTERM, signal.SIGKILL) if kill else (signal.SIGTERM,):
        _cleanup_call(lambda signum=signum: _send(child.pid, signum), errors)
        if _ended_within(child, time.monotonic() + grace, errors):
            return _drained(errors, escalated)
        escalated = "kill" if kill else "term"
    report_at = 0
    while _cleanup_call(lambda: _alive(child), errors, True):
        if time.monotonic() >= report_at:
            event = {
                **LABEL,
                "event": "group_not_ended",
                "pid": child.pid,
                "cleanup_error": bool(errors),
            }
            _cleanup_call(lambda event=event: notify(event), errors)
            report_at = time.monotonic() + 15
        time.sleep(0.05)
    return _drained(errors, escalated)


def _bounded(child, state, notify, deadline):
    expired = False
    while _alive(child):
        expired = time.monotonic() >= deadline
        signalled = state.received_at is not None and time.monotonic() - state.received_at >= 10
        if expired or signalled:
            return expired, _drain(child, notify)
        time.sleep(0.05)
    return expired, None


def execute(command, state, notify, deadline=None, env=None, stdout=None):
    metadata = {"deadline": False, "escalated": None, "signal": state.number}
    if state.number is not None:
        return 128 + state.number, metadata
    child = subprocess.Popen(
        command,
        start_new_session=True,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
    )
    state.child = child
    try:
        if state.number is not None:
            state.receive(state.number, None)
        if deadline is None:
            child.wait()
            _wait_group(child.pid)
        else:
            metadata["deadline"], metadata["escalated"] = _bounded(child, state, notify, deadline)
        status = child.wait()
        if getattr(state, "errors", []):
            raise state.errors[0]
        notify({**LABEL, "event": "heavy_leader_finished", "pid": child.pid, "exit_status": status})
    except BaseException:
        _drain(child, notify, kill=deadline is not None)
        raise
    finally:
        state.child = None
    metadata.update(signal=state.number, leader_pid=child.pid)
    return (128 + state.number if state.number is not None else status), metadata


def run(command, worker, notify):
    require_unix()
    if not command:
        raise ValueError("--local-heavy requires a command after --")
    with signals() as state:
        try:
            with held(state, worker, "build or check", notify):
                status, _ = execute(_priority(command, notify), state, notify)
                return status
        except InterruptedError:
            if state.number is None:
                raise
            return 128 + state.number
