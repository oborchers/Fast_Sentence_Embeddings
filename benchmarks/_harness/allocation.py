"""Run one allocation: machine check first, then the steps in order, each with its own
result folder, DONE, PID and restart guard, all under one time and money budget.

    python -m _harness.allocation begin --spec FILE --entry E --root R --allocated-at T ...
    python -m _harness.allocation run   --spec FILE --entry E --root R --allocated-at T ...
    python -m _harness.allocation collect --root R --archives DIR

Run it from the ``benchmarks`` folder (``PYTHONPATH``), like every module of the harness.

``begin`` checks the machine before anything is installed or started. A rejected machine
seals every step as unstarted and the allocation as failed. ``run`` admits each step
against the shared budget with its reserve, starts it once, and goes on to the next step
after a failure: a failed step 1 never hides step 2. SIGTERM goes to the running step and
stops every later step. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from _harness import machine, prerequisites, spec
from _harness.budget import Budget, select_billing
from _harness.collect import collect, verify
from _harness.records import append, atomic, document, read_rows, seal, seal_unstarted

RESTART_STATUS = 77


class Allocation:
    def __init__(self, root, entry, definition, allocated_at, budget):
        self.root, self.entry, self.definition = root, entry, definition
        self.allocated_at, self.budget = allocated_at, budget
        self.record = root / "allocation-results"
        self.stopping, self.active = False, None

    def results(self, step):
        return self.root / f"{step['name']}-results"

    def begin(self, rehearsal, work):
        """Create the record and check the machine; the check result, accepted or not."""
        self.record.mkdir(parents=True, exist_ok=False)
        expected = self.definition["rehearsal_machine_check" if rehearsal else "machine_check"]
        checked = machine.check(expected, work)
        atomic(self.record / "machine-check.json", {"rehearsal": rehearsal, **checked})
        atomic(
            self.record / "allocation.json",
            {
                "definition": self.definition,
                "allocated_at": self.allocated_at,
                "billing_profile": self.budget.profile,
                "billing": self.budget.billing,
                "budget_ceiling_seconds": self.budget.ceiling_seconds,
                "collection_reserve_seconds": self.budget.reserve_seconds,
            },
        )
        if checked["status"] == "rejected":
            self.refuse("machine check rejected: " + "; ".join(checked["deviations"]))
        return checked

    def refuse(self, reason):
        """No step starts: every step folder is sealed as unstarted, the allocation fails."""
        for step in self.definition["steps"]:
            self.skip(step, reason)
        self.finish(1)

    def step_row(self, step, status, exit_status, reason=None, evidence=None):
        append(
            self.record / "steps.jsonl",
            {
                "step": step["name"],
                "status": status,
                "exit_status": exit_status,
                "reason": reason,
                "evidence": evidence,
                "results": self.results(step).name,
                "at_unix": time.time(),
                "elapsed_seconds": self.budget.elapsed(),
            },
        )

    def claim(self, step):
        """The restart guard of one step: the first start claims it, any later one is refused."""
        state = self.root / "step-state"  # not <step>-state: the bootstrap owns that
        state.mkdir(exist_ok=True)
        try:
            with (state / step["name"]).open("x") as stream:
                stream.write(f"{time.time()}\n")
        except FileExistsError:
            return False
        return True

    def start(self, step, command):
        logs = self.root / f"{step['name']}-logs"
        logs.mkdir(exist_ok=True)
        env = {**os.environ, "MEASURE_BILLING_PROFILE": self.budget.profile or ""}
        process = subprocess.Popen(command, env=env)
        self.active = process  # registered at once: a stop signal reaches it from here on
        try:
            (logs / "measurement.pid").write_text(f"{process.pid}\n")
        except OSError:
            process.kill()
            process.wait()
            raise
        return process

    def step_values(self, values, step):
        """The fields of one step: its own folders and the interpreter of its environment."""
        name = step["name"]
        python = self.root / f"{self.entry}-env" / step["environment"] / "bin/python"
        return {
            **values,
            "root": str(self.root),
            "allocated_at": str(self.allocated_at),
            "results": str(self.results(step)),
            "work": str(self.root / f"{name}-work"),
            "python": str(python),
            "remaining_seconds": str(self.budget.remaining()),
        }

    def prepare_step(self, step, values):
        """Resolve evidence and command before the final admission check."""
        self.check_budget()
        reason = prerequisites.refusal(self, step, values)
        if reason:
            return None, None, reason
        try:
            context = prerequisites.identity(self, values)
        except (OSError, ValueError) as error:
            return None, None, f"prerequisite identity unavailable: {error}"
        try:
            command = spec.render(step, self.step_values(values, step))
        except ValueError as error:
            return None, None, f"invalid step: {error}"
        return command, context, None

    def admission_refusal(self, step):
        if self.stopping:
            return "stop signal before the step"
        if not self.budget.admit(step["estimate_seconds"]):
            return "time or money budget with reserve exhausted"
        return prerequisites.deadline_refusal(self, step)

    def run_step(self, step, values):
        """Run one step to its end; the exit status of the step."""
        command, context, reason = self.prepare_step(step, values)
        reason = reason or self.admission_refusal(step)
        if reason:
            return self.skip(step, reason)
        if not self.claim(step):
            self.step_row(step, "restart_refused", RESTART_STATUS, "step already started once")
            return RESTART_STATUS
        if self.results(step).exists():
            self.step_row(
                step, "not_started", 1, "result folder already exists; evidence not reused"
            )
            return 1
        try:
            process = self.start(step, command)
        except OSError as error:
            return self.skip(step, f"step could not start: {error!r}")
        if self.stopping:  # a signal between the admission and the registration
            process.send_signal(signal.SIGTERM)
        code = process.wait()
        return self.completed_step(step, code, context)

    def completed_step(self, step, code, context):
        results = self.results(step)
        sealed = (results / "DONE").exists()
        if not sealed:
            seal_unstarted(results, code or 1, f"step ended without DONE (exit {code})")
        status, reason = prerequisites.completion_status(results, code)
        evidence = None
        if sealed and status == 0 and context is not None:
            try:
                evidence = prerequisites.receipt(results, context)
            except (OSError, ValueError, KeyError, TypeError) as error:
                status = 1
                reason = f"invalid prerequisite seal: {error}"
        self.step_row(step, "finished" if sealed else "unsealed", status, reason, evidence)
        return status

    def skip(self, step, reason):
        if not self.results(step).exists():
            seal_unstarted(self.results(step), 1, reason)
        else:
            reason += "; existing result folder preserved"
        self.step_row(step, "not_started", 1, reason)
        return 1

    def forward(self, signum, _frame):
        self.stopping = True
        if self.active is not None and self.active.poll() is None:
            self.active.send_signal(signum)

    def run(self, values):
        if not self.record.is_dir() or (self.record / "DONE").exists():
            raise RuntimeError("run needs a begun allocation that is not sealed yet")
        self.check_budget()
        previous = {s: signal.signal(s, self.forward) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            codes = [self.run_step(step, values) for step in self.definition["steps"]]
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        return self.finish(int(any(codes)))

    def check_budget(self):
        """Refuse a changed profile, limit or clock after the machine check."""
        saved = document(self.record / "allocation.json")
        expected = {
            "definition": self.definition,
            "allocated_at": self.allocated_at,
            "billing_profile": self.budget.profile,
            "billing": self.budget.billing,
            "budget_ceiling_seconds": self.budget.ceiling_seconds,
            "collection_reserve_seconds": self.budget.reserve_seconds,
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("allocation budget differs from the machine-check budget")

    def finish(self, code):
        rows = read_rows(self.record, "steps.jsonl")
        atomic(
            self.record / "progress.json",
            {"stage": "terminal", "finished": len(self.definition["steps"]), "exit_status": code},
        )
        seal(self.record, code, len(rows) or len(self.definition["steps"]))
        return code

    def folders(self):
        return [self.record, *(self.results(step) for step in self.definition["steps"])]


def budget_of(study_file, pins_file, profile, allocated_at, study=None):
    contract, pins = json.loads(study_file.read_text()), json.loads(pins_file.read_text())
    resolved = select_billing(contract, profile, study)
    return Budget.from_contract({**resolved, "machine": pins["machine"]}, allocated_at)


def collect_all(root, archives):
    """One verified archive per folder of an allocation, steps and record alike."""
    archives.mkdir(parents=True, exist_ok=False)
    declared = document(root / "allocation-results" / "allocation.json")["definition"]["steps"]
    names = ["allocation-results", *(f"{step['name']}-results" for step in declared)]
    for name in names:
        verify(root / name)
    return [collect(root / name, archives / f"{name}.tar.gz") for name in names]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("begin", "run", "collect"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--entry", choices=("speed", "quality"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allocated-at", type=float)
    parser.add_argument("--study-file", type=Path)
    parser.add_argument("--pins-file", type=Path)
    parser.add_argument("--billing-profile")
    parser.add_argument("--study")
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--param", action="append", default=[], help="NAME=VALUE for step fields")
    return parser


def _values(args, definition):
    """Fields every step may use: the source folder, --param values and
    the interpreter of each environment."""
    params = dict(item.split("=", 1) for item in args.param)
    if args.study is not None:
        if "study" in params and params["study"] != args.study:
            raise ValueError("--study and --param study disagree")
        params["study"] = args.study
    values = {"source": str(Path(__file__).resolve().parents[2]), **params}
    for environment in definition["environments"]:
        python = args.root / f"{args.entry}-env" / environment["name"] / "bin/python"
        values[f"python_{environment['name']}"] = str(python)
    return values


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.action == "collect":
        print(json.dumps(collect_all(args.root, args.archives)))
        return 0
    definition = spec.load(args.spec, args.entry)
    values = _values(args, definition)
    budget = budget_of(
        args.study_file,
        args.pins_file,
        args.billing_profile,
        args.allocated_at,
        values.get("study"),
    )
    values["billing_profile"] = budget.profile or ""
    allocation = Allocation(args.root, args.entry, definition, args.allocated_at, budget)
    if args.action == "begin":
        return int(allocation.begin(args.rehearsal, args.root)["status"] == "rejected")
    return allocation.run(values)


if __name__ == "__main__":
    sys.exit(main())
