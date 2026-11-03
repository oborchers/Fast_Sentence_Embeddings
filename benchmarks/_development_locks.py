"""Mutually exclusive directory locks shared by timing and heavy Unix work."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from _development_system import LABEL, require_unix


def lock_paths():
    require_unix()
    root = Path(tempfile.gettempdir()) / f"fse-development-{os.getuid()}"
    paths = tuple(
        Path(os.environ.get(variable, str(root / name))).absolute()
        for variable, name in (("FSE_BENCH_LOCK", "bench.lock"), ("FSE_HEAVY_LOCK", "heavy.lock"))
    )
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("bench.lock and heavy.lock must have different paths")
    return paths


def busy(path):
    return path.exists() or path.is_symlink()


def _release(path, owner):
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("owned development lock disappeared or changed type")
    file = path / "owner"
    if file.is_symlink() or json.loads(file.read_text()) != owner:
        raise RuntimeError("development lock owner changed; refusing to remove it")
    file.unlink()
    path.rmdir()


def _claim(path, owner):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        return False
    with (path / "owner").open("x") as stream:
        json.dump(owner, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return True


@contextmanager
def coordinated(role, worker, purpose, notify, stopped=lambda: False, admitted=lambda: True):
    """Claim only our own lock, recheck its peer, and never take over any owner."""
    bench, heavy = lock_paths()
    path, peer = (bench, heavy) if role == "timing" else (heavy, bench)
    if role not in ("timing", "heavy") or not worker or not purpose:
        raise ValueError("a timing/heavy role, worker and purpose are required")
    owner = dict(worker=worker, pid=os.getpid(), purpose=purpose, token=uuid.uuid4().hex)
    while True:
        if stopped():
            raise InterruptedError("stopped before development lock admission")
        if not busy(peer) and admitted() and _claim(path, owner):
            if not busy(peer) and admitted():
                break
            _release(path, owner)
        notify({**LABEL, "event": "waiting_for_lock", "lock": str(path), "peer": str(peer)})
        time.sleep(1)
    try:
        notify({**LABEL, "event": "lock_acquired", "lock": str(path), "owner": owner})
        yield owner
    finally:
        _release(path, owner)
        notify({**LABEL, "event": "lock_released", "lock": str(path), "owner": owner})


def heavy_command(command, worker, notify):
    require_unix()
    if not command:
        raise ValueError("--local-heavy requires a command after --")
    with coordinated("heavy", worker, "build or check", notify):
        # nice is optional on Unix. Absence is visible and does not prevent operation.
        import shutil

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
        child = subprocess.Popen([nice, "-n", str(increment), *command] if nice else command)
        try:
            status = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    return status
