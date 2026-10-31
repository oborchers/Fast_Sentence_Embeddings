"""Independent supervision of a gate subprocess and its process group.

The owner alone holds a pipe's write end. EOF therefore detects even SIGKILL of
the owner without relying on Python code in a hung encoder. This small stdlib-only
supervisor stays outside the job's group and kills that group on owner loss,
deadline, termination or normal completion (including leftover descendants).
"""

from __future__ import annotations

import math
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path

POLL_SECONDS = 0.25
TERM_GRACE = 0.25
CLEANUP_SECONDS = 2.0


class TerminatedError(KeyboardInterrupt):
    """Controlled SIGTERM, so caller finally blocks can retain evidence and settle budget."""


def _terminated(_number, _frame):
    raise TerminatedError("gate owner received SIGTERM")


@contextmanager
def controlled_termination():
    before = signal.signal(signal.SIGTERM, _terminated)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, before)


def _close(fd: int) -> None:
    if fd >= 0:
        os.close(fd)


def _finish(supervisor) -> None:
    try:
        supervisor.wait(timeout=CLEANUP_SECONDS)
    except subprocess.TimeoutExpired:
        supervisor.terminate()
        try:
            supervisor.wait(timeout=CLEANUP_SECONDS)
        except subprocess.TimeoutExpired:
            supervisor.kill()
            supervisor.wait(timeout=CLEANUP_SECONDS)


def _deadline(timeout: float | None, deadline: float | None) -> float:
    if deadline is not None:
        if timeout is not None or not math.isfinite(deadline):
            raise ValueError("provide one finite deadline or a positive timeout")
        return deadline
    if timeout is None or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("subprocess timeout must be finite and positive")
    return time.monotonic() + timeout


def run_job(command, log, timeout: float | None = None, *, deadline: float | None = None) -> int:
    """Start no job without time remaining; controlled and hard owner death are covered."""
    deadline = _deadline(timeout, deadline)
    if time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired(command, 0)
    read_fd, owner_fd = os.pipe()
    supervisor = None
    try:
        with controlled_termination():
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, 0)
            supervisor = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(read_fd),
                    str(deadline),
                    *command,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(read_fd,),
                start_new_session=True,
            )
            _close(read_fd)
            read_fd = -1
            return supervisor.wait(timeout=max(0, deadline - time.monotonic()) + CLEANUP_SECONDS)
    finally:
        # Only the original owner has this descriptor. Do not pass it to any job.
        _close(owner_fd)
        _close(read_fd)
        if supervisor is not None:
            _finish(supervisor)


def _owner_present(fd: int, timeout: float) -> bool:
    ready, _, _ = select.select([fd], [], [], timeout)
    return not ready or bool(os.read(fd, 1))


def _kill_group(child) -> None:
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        child.wait(timeout=TERM_GRACE)
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=CLEANUP_SECONDS)


def supervise(owner_fd: int, deadline: float, command: list[str]) -> int:
    """No fse import, GIL dependence on the encoder, or unbounded wait in the monitor."""
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        return 124
    if not _owner_present(owner_fd, 0):
        return 125  # owner died during supervisor startup; do not launch work
    with controlled_termination():
        child = subprocess.Popen(command, start_new_session=True)
        try:
            return _watch(child, owner_fd, deadline)
        finally:
            _kill_group(child)


def _watch(child, owner_fd: int, deadline: float) -> int:
    while True:
        code = child.poll()
        if code is not None:
            return code if code >= 0 else 128 - code
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("gate supervisor: deadline expired", file=sys.stderr, flush=True)
            return 124
        if not _owner_present(owner_fd, min(POLL_SECONDS, remaining)):
            print("gate supervisor: owner exited", file=sys.stderr, flush=True)
            return 125


if __name__ == "__main__":
    raise SystemExit(supervise(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]))
