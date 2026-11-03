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

    def receive(self, signum, _frame):
        self.number = self.number or signum
        if self.child is not None:
            _send(self.child.pid, signum)


@contextmanager
def _handled():
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


def _execute(command, state, notify):
    if state.number is not None:
        return 128 + state.number
    child = subprocess.Popen(command, start_new_session=True)
    state.child = child
    if state.number is not None:
        state.receive(state.number, None)
    try:
        status = child.wait()
        notify({**LABEL, "event": "heavy_leader_finished", "pid": child.pid, "exit_status": status})
        _wait_group(child.pid)
    except BaseException:
        _send(child.pid, signal.SIGTERM)
        child.wait()
        _wait_group(child.pid)
        raise
    finally:
        state.child = None
    return 128 + state.number if state.number is not None else status


def run(command, worker, notify):
    require_unix()
    if not command:
        raise ValueError("--local-heavy requires a command after --")
    with _handled() as state:
        try:
            with coordinated(
                "heavy", worker, "build or check", notify, lambda: state.number is not None
            ):
                return _execute(_priority(command, notify), state, notify)
        except InterruptedError:
            if state.number is None:
                raise
            return 128 + state.number
