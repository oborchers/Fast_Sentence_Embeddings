"""Explicit installed interpreters and shared child jobs for both measurement entries."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _development_system import LABEL, configure_child, require_unix
from _measurement import atomic, document, package_identity, sha256

ENTRY = Path(__file__).with_name("speed.py").resolve()


def interpreter(value):
    if not value:
        raise ValueError("local development requires explicit base and candidate interpreters")
    path = Path(value).absolute()
    if path.is_dir():
        path = path / "bin/python"
    # Do not resolve the interpreter symlink: the venv directory selects its installation.
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"not an executable Python interpreter: {path}")
    return str(path)


def launch(args, job, python, tag, timed=False):
    job = {**job, **LABEL, "receipt": str(args.results / (tag + ".json")), "work": str(args.work)}
    file = args.results / (tag + ".job.json")
    atomic(file, job)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["FSE_DEVELOPMENT_TIMED"] = "1" if timed else "0"
    command = [interpreter(python), str(ENTRY), "--development-job", str(file)]
    nice = shutil.which("nice")
    if not timed and nice:
        command = [nice, "-n", "10", *command]
    with file.with_suffix(".log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if code:
        _failed(job, code, file.with_suffix(".log"))
    receipt = document(Path(job["receipt"]))
    if receipt.get("mode") != "local-development" or receipt.get("publishable") is not False:
        raise ValueError("child omitted the local development result label")
    return receipt


def _failed(job, code, log):
    receipt = Path(job["receipt"])
    if receipt.is_file():
        failure = document(receipt)
        if failure.get("error") == "session_expired":
            raise TimeoutError(failure["reason"])
    raise RuntimeError(f"development child exited with status {code}; see {log}")


def identity():
    return {
        "python": sys.version,
        "executable": sys.executable,
        "fse": package_identity("fse"),
        "dependencies": dict(
            sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        ),
    }


def child(file):
    require_unix()
    job = document(file)
    workers = job.get("workers", 1)
    configure_child(workers)
    started = time.monotonic()
    before = identity()
    if job["action"] == "prepare":
        from _development_data import prepare_child

        result = prepare_child(job)
    elif job["action"] == "quality":
        from _quality.development_child import export

        result = export(job)
    elif job["action"] == "measure":
        from _development_measure import measure

        try:
            result = measure(job)
        except TimeoutError as error:
            atomic(
                Path(job["receipt"]), {**LABEL, "error": "session_expired", "reason": str(error)}
            )
            raise
    elif job["action"] == "pooling-input":
        from _development_stages import prepare_batch

        result = {"batch": prepare_batch(job)}
    else:
        raise ValueError("unknown development child action")
    if identity() != before:
        raise ValueError("installed build changed during development child execution")
    result["active_seconds"] = time.monotonic() - started - result.pop("wait_seconds", 0.0)
    atomic(Path(job["receipt"]), {**result, **LABEL, "identity": before})


def checked(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError(f"prepared development input changed: {path}")
    return path
