"""Real blocked-process cleanup, including SIGKILL of the owner, without benchmarks."""

import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from support.waiting import until

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "mlx_gate_process.py").exists() or sys.platform == "win32",
    reason="gate process supervision requires POSIX process groups",
)

# The SIGUSR1 probe reports the actual interrupted Python frame. It proves that
# each process reached block(), whose only operation is the blocking recv call.
# Readiness alone, or a sleep after startup, would not establish that state.
BLOCKED = """
import json, os, signal, socket, subprocess, sys
connection = socket.create_connection(('127.0.0.1', int(sys.argv[1])))
def probe(_number, frame):
    connection.sendall((frame.f_code.co_name + '\\n').encode())
signal.signal(signal.SIGUSR1, probe)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
def block():
    connection.recv(1)
if sys.argv[2] == 'leader':
    subprocess.Popen([sys.executable, __file__, sys.argv[1], 'descendant'])
connection.sendall((json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}) + '\\n').encode())
block()
"""


def _line(connection) -> bytes:
    out = bytearray()
    while not out.endswith(b"\n"):
        value = connection.recv(1)
        assert value, "process exited before its synchronization response"
        out.extend(value)
    return bytes(out)


def _blocked(connection, pid: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        os.kill(pid, signal.SIGUSR1)
        if _line(connection) == b"block\n":
            return
    raise AssertionError("process did not reach the blocking recv")


def _stopped(pid: int) -> bool:
    found = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
    return found.returncode != 0 or found.stdout.strip().startswith("Z")


def _owner_program() -> str:
    return """
import sys
from mlx_gate_process import run_job
command = [sys.executable, sys.argv[1], sys.argv[2], 'leader']
with open(sys.argv[3], 'w') as log:
    raise SystemExit(run_job(command, log, float(sys.argv[4])))
"""


@pytest.mark.parametrize("termination", ["SIGTERM", "SIGKILL", "deadline"])
def test_owner_termination_or_deadline_stops_blocked_group_and_descendant(tmp_path, termination):
    program = tmp_path / "blocked.py"
    program.write_text(BLOCKED, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(BENCHMARKS))
    processes, connections = [], []
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(2)
        server.settimeout(15)
        duration = 10 if termination == "deadline" else 60
        owner = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _owner_program(),
                str(program),
                str(server.getsockname()[1]),
                str(tmp_path / "child.log"),
                str(duration),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(2):
                connection, _address = server.accept()
                connections.append(connection)
                connection.settimeout(15)
                processes.append(json.loads(_line(connection)))
            assert processes[0]["pgid"] == processes[1]["pgid"]
            for connection, process in zip(connections, processes, strict=True):
                _blocked(connection, process["pid"])
            assert owner.poll() is None
            if termination != "deadline":
                owner.send_signal(getattr(signal, termination))
            owner.wait(timeout=15)
            for connection in connections:
                assert connection.recv(1) == b""  # cleanup, not a guessed elapsed delay
            assert until(lambda: all(_stopped(p["pid"]) for p in processes))
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=5)
            for connection in connections:
                connection.close()


def test_exhausted_timeout_starts_nothing(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    process = importlib.import_module("mlx_gate_process")

    def forbidden(*_a, **_k):
        pytest.fail("an exhausted timeout must not launch a supervisor or command")

    monkeypatch.setattr(process.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="finite and positive"):
        process.run_job(["unused"], None, 0)
