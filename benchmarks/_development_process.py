"""Explicit installed interpreters and shared child jobs for both measurement entries."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _development_budget import WallBudgetExpiredError, check, check_args
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
    check_args(args)
    job = {
        **job,
        "wall_deadline": getattr(args, "wall_deadline", None),
        **LABEL,
        "receipt": str(args.results / (tag + ".json")),
        "work": str(args.work),
    }
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
        if failure.get("error") == "wall_budget":
            raise WallBudgetExpiredError(failure["reason"])
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
    try:
        check(job.get("wall_deadline"))
    except WallBudgetExpiredError as error:
        atomic(Path(job["receipt"]), {**LABEL, "error": "wall_budget", "reason": str(error)})
        raise
    started = time.monotonic()
    before = identity()
    try:
        check(job.get("wall_deadline"))
        result = _action(job)
    except TimeoutError as error:
        atomic(
            Path(job["receipt"]),
            {
                **LABEL,
                "error": "wall_budget"
                if isinstance(error, WallBudgetExpiredError)
                else "session_expired",
                "reason": str(error),
            },
        )
        raise
    if identity() != before:
        raise ValueError("installed build changed during development child execution")
    result["active_seconds"] = time.monotonic() - started - result.pop("wait_seconds", 0.0)
    atomic(Path(job["receipt"]), {**result, **LABEL, "identity": before})


def _head_profile(job):
    if job["profile"] in ("linear-heads-multi@1", "linear-heads-scores@1"):
        from _development_heads_multi import profile
    else:
        from _development_heads import profile
    return profile(job)


def _action(job):
    action = job["action"]
    if action == "prepare":
        from _development_data import prepare_child

        return prepare_child(job)
    if action == "probe":
        from _development_preflight import probe

        return probe(job)
    if action == "quality":
        from _quality.development_child import export

        return export(job)
    if action == "prefilter-count":
        from _quality.prefilter_child import count_job

        return count_job(job)
    if action == "measure":
        from _development_measure import measure

        return measure(job)
    return _other_action(job)


def _other_action(job):
    action = job["action"]
    if action == "identity":
        from _development_identity_run import measure

        return measure(job)
    if action == "profile-head":
        return _head_profile(job)
    if action == "binary-recall":
        from _quality.binary_child import count_job as binary_job

        return binary_job(job)
    if action == "pooling-input":
        from _development_stages import prepare_batch

        return {"batch": prepare_batch(job)}
    if action == "filter-memory":
        from _development_filter_memory import profile

        return profile(job)
    raise ValueError("unknown development child action")


def checked(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError(f"prepared development input changed: {path}")
    return path
