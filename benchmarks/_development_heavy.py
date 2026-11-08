"""Signal-forwarding Unix heavy jobs that own their lock until the group finishes."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager, suppress

from _development_locks import coordinated
from _development_system import LABEL, require_unix
from _harness.budget import check


def _send(group, signum):
    with suppress(ProcessLookupError):
        os.killpg(group, signum)


def _wait_group(group):
    while True:
        _reap(group)
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
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

    def receive(self, signum, _frame):
        self.received_at = self.received_at or time.monotonic()
        self.number = self.number or signum
        if self.child is not None:
            _send(self.child.pid, signum)


@contextmanager
def signals():
    state = _Signals()
    previous = {n: signal.signal(n, state.receive) for n in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield state
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


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


def _alive(child):
    child.poll()
    _reap(child.pid)
    try:
        os.killpg(child.pid, 0)
    except ProcessLookupError:
        return False
    return True


def _drain(child, notify, grace=10.0):
    escalated = "term"
    for signum in (signal.SIGTERM, signal.SIGKILL):
        _send(child.pid, signum)
        end = time.monotonic() + grace
        while _alive(child) and time.monotonic() < end:
            time.sleep(0.05)
        if not _alive(child):
            return escalated
        escalated = "kill"
    report_at = 0
    while _alive(child):
        if time.monotonic() >= report_at:
            notify({**LABEL, "event": "group_not_ended", "pid": child.pid})
            report_at = time.monotonic() + 15
        time.sleep(0.05)
    return escalated


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
        command, start_new_session=True, env=env, stdout=stdout, stderr=subprocess.STDOUT
    )
    state.child = child
    if state.number is not None:
        state.receive(state.number, None)
    try:
        if deadline is None:
            child.wait()
            _wait_group(child.pid)
        else:
            metadata["deadline"], metadata["escalated"] = _bounded(child, state, notify, deadline)
        status = child.wait()
        notify({**LABEL, "event": "heavy_leader_finished", "pid": child.pid, "exit_status": status})
    except BaseException:
        if deadline is None:
            _send(child.pid, signal.SIGTERM)
            child.wait()
            _wait_group(child.pid)
        else:
            _drain(child, notify)
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
