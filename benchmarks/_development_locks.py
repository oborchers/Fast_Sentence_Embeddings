"""Mutually exclusive directory locks shared by timing and heavy Unix work."""

from __future__ import annotations

import errno
import json
import math
import os
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from _development_queue import Waiter, validate_paths
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
    validate_paths(paths)
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
    created, written = path.lstat(), None
    try:
        with (path / "owner").open("x") as stream:
            written = os.fstat(stream.fileno())
            json.dump(owner, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        _rollback_claim(path, created, written)
        raise
    return True


def _same_file(path, expected):
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino)


def _rollback_claim(path, created, written):
    if not _same_file(path, created):
        return
    owner = path / "owner"
    if written is not None and _same_file(owner, written):
        owner.unlink()
    if _same_file(path, created):
        try:
            path.rmdir()
        except OSError as error:
            if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise


def _holder(path):
    try:
        fd = os.open(path / "owner", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None
            data = os.read(fd, 65537)
            return json.loads(data) if len(data) <= 65536 else None
        finally:
            os.close(fd)
    except (OSError, ValueError):
        return None


def _check_stopped(stopped):
    if stopped():
        raise InterruptedError("stopped before development lock admission")


def _waiting(path, peer, position, notify):
    place, count = position if position is not None else (None, None)
    notify(
        {
            **LABEL,
            "event": "waiting_for_lock",
            "lock": str(path),
            "peer": str(peer),
            "position": place,
            "waiting": count,
            "holder": _holder(path),
            "peer_holder": _holder(peer),
        }
    )


@contextmanager
def _claimed(path, peer, owner, admitted, notify):
    acquired = _claim(path, owner)
    accepted = False
    try:
        accepted = acquired and not busy(peer) and admitted()
        yield accepted
    finally:
        if acquired:
            _release(path, owner)
            if accepted:
                notify({**LABEL, "event": "lock_released", "lock": str(path), "owner": owner})


@contextmanager
def coordinated(
    role, worker, purpose, notify, stopped=lambda: False, admitted=lambda: True, *, poll=1.0
):
    """Admit the first registered waiter; never take over an existing lock owner."""
    if not math.isfinite(poll) or poll <= 0:
        raise ValueError("poll must be finite and positive")
    bench, heavy = lock_paths()
    path, peer = (bench, heavy) if role == "timing" else (heavy, bench)
    if role not in ("timing", "heavy") or not worker or not purpose:
        raise ValueError("a timing/heavy role, worker and purpose are required")
    owner = dict(worker=worker, pid=os.getpid(), purpose=purpose, token=uuid.uuid4().hex)
    waiter = Waiter(path, owner)
    try:
        while True:
            _check_stopped(stopped)
            position = waiter.position()
            _check_stopped(stopped)
            if position is not None and position[0] == 1 and not busy(peer) and admitted():
                with _claimed(path, peer, owner, admitted, notify) as acquired:
                    if acquired:
                        waiter.close()
                        notify(
                            {**LABEL, "event": "lock_acquired", "lock": str(path), "owner": owner}
                        )
                        _check_stopped(stopped)
                        yield owner
                        break
            _waiting(path, peer, position, notify)
            time.sleep(poll)
    finally:
        waiter.close()


def heavy_command(command, worker, notify):
    from _development_heavy import run

    return run(command, worker, notify)
