"""Profile an addressed child once, under one heavy lock and its own wall deadline."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from _development_heavy import _priority, execute, held, signals
from _development_profile_origin import HERE, check_work_paths, resolve, verify_again
from _development_profile_probe import access_program
from _development_records import LocalLedger
from _development_system import LABEL, machine, mark_diagnostic, mark_functional
from _harness.budget import WallBudgetExpiredError, check
from _harness.ledger import beating
from _harness.profile import identity, lock_packages, rewrite, same_identity, sources, validate
from _harness.profile_tools import LIMITS, SOURCES, command, coverage
from _harness.records import append, atomic, document, seal_unstarted, sha256
from _measurement_setup import outside_checkout

ENTRY = HERE / "speed.py"


class Runner:
    def __init__(self, spec, state, deadline, prepare=None):
        self.spec, self.state, self.deadline = spec, state, deadline
        self.root = Path(spec["output"])
        self.probes = 0
        self.prepare = prepare
        self.tool_executable = None
        self.open_evidence = []

    def notify(self, value):
        append(self.root / "events.jsonl", value)

    def launch(self, target, name):
        check(self.deadline - 30)
        if self.state.number is not None:
            raise InterruptedError("profile interrupted")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            FSE_DEVELOPMENT_TIMED="0",
            MEASURE_WORK=self.spec["work"],
            HF_HUB_OFFLINE="1",
        )
        with (self.root / (name + ".log")).open("x") as log:
            status, meta = execute(
                _priority(target, self.notify),
                self.state,
                self.notify,
                deadline=self.deadline - 30,
                env=env,
                stdout=log,
            )
        atomic(self.root / (name + "-execution.json"), {"status": status, **meta})
        if meta["deadline"]:
            raise WallBudgetExpiredError("profile process group exceeded its deadline")
        if self.state.number is not None:
            raise InterruptedError("profile interrupted")
        if status and name == "profile":
            self.child_failure()
        if status or meta["escalated"]:
            raise RuntimeError(
                f"{name} exited with status {status}; escalation {meta['escalated']}"
            )
        return meta

    def child_failure(self):
        job = document(self.root / "job.json")
        receipt = Path(job["receipt"])
        if receipt.is_file():
            error = document(receipt)
            if error.get("error") == "wall_budget":
                raise WallBudgetExpiredError(error["reason"])

    def probe(self, action, **extra):
        self.probes += 1
        name = f"{action}-{self.probes}"
        receipt = self.root / (name + ".json")
        file = self.root / (name + ".job.json")
        atomic(file, {"action": action, "receipt": str(receipt), **extra})
        self.launch([self.spec["python"], str(ENTRY), "--profile-probe", str(file)], name)
        return document(receipt)

    def profiler(self):
        if self.spec["tool"] == "memray":
            value = self.probe("version")
            value["sha256"] = sha256(Path(value["path"]))
            return value
        executable = self.tool_executable or shutil.which(self.spec["tool"])
        if executable is None:
            raise ValueError("profile tool is not installed")
        self.launch([executable, "--version"], "version")
        return {
            "path": executable,
            "sha256": sha256(Path(executable)),
            "version": (self.root / "version.log").read_text().strip(),
        }

    def inspect(self, folder, receipt, meta, profiler):
        try:
            if self.spec["tool"] == "memray":
                # Read in the profile interpreter, without controller dependencies.
                value = self.probe("memray-metadata", profile=str(folder / "mem.bin"))
                atomic(folder / "memray-metadata.json", value)
            result = coverage(self.spec, folder, receipt, meta["leader_pid"], profiler)
        except (WallBudgetExpiredError, InterruptedError):
            raise
        except Exception as error:
            tool = self.spec["tool"]
            atomic(
                folder / "coverage.json",
                {
                    "requested": self.spec["requested"],
                    "observed": None,
                    "assured": {
                        "limits": LIMITS[tool],
                        "source": SOURCES[tool],
                        "version": profiler["version"],
                    },
                    "valid": False,
                    "reason": "unparsed",
                    "error": str(error),
                },
            )
            raise ValueError("unparsed profiler coverage") from error
        atomic(folder / "coverage.json", result)
        if not result["valid"]:
            raise ValueError("profile coverage does not prove child attribution and data")
        return result

    def access(self, executable, profiler):
        folder = self.root / "access"
        folder.mkdir()
        file, receipt = folder / "check.py", folder / "receipt.json"
        access_program(file, receipt)
        target = [self.spec["python"], "-I", str(file)]
        try:
            meta = self.launch(command(self.spec, executable, folder, target), "access")
            result = self.inspect(folder, document(receipt), meta, profiler)
        except Exception as error:
            atomic(
                self.root / "check.json",
                {
                    "requested": self.spec["requested"],
                    "valid": False,
                    "reason": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        atomic(self.root / "check.json", result)
        return result


def _operations(runner, check_only, result):
    spec = runner.spec
    result["phase"] = "preflight"
    origin = None if check_only else resolve(spec)
    extra = lock_packages(spec["extra_lock"]) if spec["tool"] == "memray" else {}
    with held(
        runner.state, spec["name"], "profile " + spec["name"], runner.notify, runner.deadline
    ):
        if runner.prepare is not None:
            runner.prepare(runner, origin)
            environment = document(runner.root / "environment.json")
            atomic(runner.root / "environment.json", {**environment, "profile": spec})
        profiler = runner.profiler()
        if spec["extra_lock"] is not None:
            profiler.update(extra_lock_sha256=sha256(Path(spec["extra_lock"])), packages=extra)
        atomic(runner.root / "profiler.json", profiler)
        before = None
        if origin is not None:
            before = identity(runner.probe("identity"), "development")
            same_identity(origin["identity"], before, extra)
        if spec["tool"] == "memray" and profiler["version"] != extra["memray"]:
            raise ValueError("installed memray differs from profile lock")
        result["access_coverage"] = runner.access(profiler["path"], profiler)
        if check_only:
            return
        result["phase"] = "execution"
        job = rewrite(spec, origin["job"], runner.deadline, origin["job_sha256"])
        if spec["entry"] == "speed":
            from _speed_profile import rebuild, verify_rebuilt

            rebuild(spec, origin, job, runner.launch)
            verify_rebuilt(spec, job)
        file = runner.root / "job.json"
        atomic(file, job)
        flag = "--development-job" if spec["entry"] == "development" else "--child-job"
        target = [spec["python"], str(ENTRY), flag, str(file)]
        meta = runner.launch(command(spec, profiler["path"], runner.root, target), "profile")
        receipt = document(Path(job["receipt"]))
        current = identity(receipt, spec["entry"])
        same_identity(origin["identity"], current, extra)
        verify_again(spec, origin)
        if spec["entry"] == "speed":
            verify_rebuilt(spec, job)
        result["coverage"] = runner.inspect(runner.root, receipt, meta, profiler)
        result["provenance"] = {
            "origin": origin["identity"],
            "before": before,
            "after": current,
            "nominal_arm": spec["nominal_arm"],
            "job_sha256": origin["job_sha256"],
        }


def _attempt(runner, check_only):
    result = {
        "status": "valid",
        "causes": [],
        "coverage": None,
        "access_coverage": None,
        "provenance": None,
    }
    try:
        _operations(runner, check_only, result)
        if runner.prepare is not None:
            result["open_evidence"] = runner.open_evidence
            if runner.open_evidence:
                result["status"] = "incomplete_evidence"
    except (Exception, KeyboardInterrupt) as error:
        result["causes"].append({"type": type(error).__name__, "reason": str(error)})
        if isinstance(error, WallBudgetExpiredError):
            result["status"] = "inconclusive_budget"
        else:
            result["status"] = "refused" if result.get("phase") == "preflight" else "failed"
    return _terminal(result, runner.state, runner.deadline)


def _terminal(result, state, deadline):
    if state.number is not None:
        result["status"] = "interrupted"
        cause = {"signal": state.number}
        if cause not in result["causes"]:
            result["causes"].append(cause)
    elif time.monotonic() >= deadline - 30:
        result["status"] = "inconclusive_budget"
        cause = {"type": "WallBudgetExpiredError", "reason": "profile admission deadline ended"}
        if cause not in result["causes"]:
            result["causes"].append(cause)
    return result


def run(file, check_only=False, *, prepare=None):
    started = time.monotonic()
    spec = validate(document(file), check_only)
    for key in ("output", "work"):
        outside_checkout(Path(spec[key]))
    if not check_only:
        check_work_paths(spec)
    mark_functional()
    mark_diagnostic()
    root, work = Path(spec["output"]), Path(spec["work"])
    deadline = started + spec["wall_time_seconds"]
    with signals() as state:
        root.mkdir(parents=True)
        try:
            work.mkdir(parents=True)
            cell = {
                "id": spec["name"],
                "estimated_seconds": spec["wall_time_seconds"],
                **{k: spec[k] for k in ("entry", "job", "tool", "action")},
            }
            contract = {
                "format": "profile@1",
                **LABEL,
                "functional_only": True,
                "wall_time_budget_seconds": spec["wall_time_seconds"],
                "declared_estimated_seconds": None,
                "cells": [cell],
            }
            atomic(
                root / "environment.json",
                {
                    **LABEL,
                    "machine": machine(),
                    "contract": contract,
                    "profile": spec,
                    "measurement_source_sha256": sources(HERE),
                },
            )
            ledger = LocalLedger(root, contract, time.time())
            ledger.wall_deadline = deadline
            with beating(ledger.progress):
                result = _attempt(Runner(spec, state, deadline, prepare), check_only)
            state.finish()
            result = _terminal(result, state, deadline)
            ledger.result(cell, started, **LABEL, decision="diagnostic_profile", **result)
            return ledger.done()
        except BaseException as error:
            seal_unstarted(root, 1, repr(error))
            raise
