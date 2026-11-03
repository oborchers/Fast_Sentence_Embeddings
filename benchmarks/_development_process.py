"""Explicit installed interpreters and shared child jobs for both measurement entries."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
        raise RuntimeError(
            f"development child exited with status {code}; see {file.with_suffix('.log')}"
        )
    receipt = document(Path(job["receipt"]))
    if receipt.get("mode") != "local-development" or receipt.get("publishable") is not False:
        raise ValueError("child omitted the local development result label")
    return receipt


def identity():
    return {"python": sys.version, "executable": sys.executable, "fse": package_identity("fse")}


def child(file):
    require_unix()
    job = document(file)
    workers = job.get("workers", 1)
    configure_child(workers)
    before = identity()
    if job["action"] == "prepare":
        from _development_data import prepare_child

        result = prepare_child(job)
    elif job["action"] == "quality":
        from _quality.development_child import export

        result = export(job)
    else:
        raise ValueError("unknown development child action")
    if identity() != before:
        raise ValueError("installed build changed during development child execution")
    atomic(Path(job["receipt"]), {**result, **LABEL, "identity": before})


def checked(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError(f"prepared development input changed: {path}")
    return path
