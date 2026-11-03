"""Explicit local diagnostics with shared records, paired children and directory locks."""

from __future__ import annotations

import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from _local_resources import MAX_LOAD
from _measurement import Ledger, append, atomic, document, heartbeat, sha256
from _measurement_setup import outside_checkout
from _speed_diagnostic_child import observations

ENTRY = Path(__file__).with_name("speed.py").resolve()


@contextmanager
def _lock(path, worker, purpose):
    """Never take over a lock, even when its recorded process no longer exists."""
    try:
        path.mkdir()
    except FileExistsError:
        yield False
        return
    owner = {"worker": worker, "pid": os.getpid(), "purpose": purpose}
    (path / "owner").write_text(json.dumps(owner) + "\n")
    try:
        yield True
    finally:
        if document(path / "owner") != owner:
            raise RuntimeError("diagnostic lock owner changed; refusing to remove it")
        (path / "owner").unlink()
        path.rmdir()


def _launch(args, cell, arm, round_index, order, functional):
    tag = f"{cell['id']}-{round_index}-{order}"
    job = {
        **cell,
        **arm,
        "kind": "diagnostic",
        "receipt": str(args.results / (tag + ".json")),
        "work": str(args.work),
        "functional_only": functional,
    }
    file = args.results / (tag + "-job.json")
    atomic(file, job)
    with (args.results / (tag + ".log")).open("x") as log:
        subprocess.run(
            [arm["python"], str(ENTRY), "--child-job", str(file)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    receipt = document(Path(job["receipt"]))
    receipt.update(arm=arm["name"], round=round_index, order=order)
    if not functional and (
        receipt["telemetry_before"]["load"][0] >= MAX_LOAD
        or receipt["telemetry_before"]["foreign_cpu_over_50"]
    ):
        raise RuntimeError("timed call violated local load or foreign-process admission")
    if not functional and cell["operation"] != "filter_memory" and receipt["seconds"] < 2:
        raise RuntimeError("measured fse operation lasted less than two seconds")
    return receipt


def _pair(args, cell, round_index, ledger, functional):
    arms = cell["arms"] if round_index % 2 == 0 else list(reversed(cell["arms"]))
    receipts = []
    for order, arm in enumerate(arms):
        while not ledger.stopping:
            if _admitted(args, cell, round_index, order, functional):
                receipt = _launch(args, cell, arm, round_index, order, functional)
                receipts.append(receipt)
                append(args.results / "operations.jsonl", {"cell_id": cell["id"], **receipt})
                break
            ledger.stage = "waiting_for_load_or_foreign_cpu"
            ledger.progress()
            time.sleep(5)
        else:
            raise RuntimeError("diagnostic stopped before the next arm")
    if receipts[0]["output"] != receipts[1]["output"]:
        raise ValueError("paired outputs are not equivalent")
    return receipts


def _locked_cell(args, cell, ledger, functional):
    while not ledger.stopping:
        if _admitted(args, cell, -1, -1, functional):
            with _lock(Path(args.bench_lock), args.worker_name, cell["id"]) as acquired:
                if acquired and _admitted(args, cell, -1, -1, functional):
                    _cell(args, cell, ledger, functional)
                    return
        ledger.current, ledger.stage = cell["id"], "waiting_for_cell_lock"
        ledger.progress()
        time.sleep(5)


def _summary(values):
    q1, _, q3 = (
        statistics.quantiles(values, n=4, method="inclusive")
        if len(values) > 1
        else (values[0], values[0], values[0])
    )
    return {"median": statistics.median(values), "q1": q1, "q3": q3, "iqr": q3 - q1}


def _admitted(args, cell, round_index, order, functional):
    observation = observations()
    heavy_lock = Path(args.bench_lock).with_name("heavy.lock")
    heavy_busy = heavy_lock.exists()
    append(
        args.results / "preflights.jsonl",
        {
            "cell_id": cell["id"],
            "round": round_index,
            "order": order,
            "load_limit_exclusive": MAX_LOAD,
            "foreign_cpu_limit_percent": 50,
            "heavy_lock": str(heavy_lock),
            "heavy_lock_busy": heavy_busy,
            "telemetry": observation,
        },
    )
    return not heavy_busy and (
        functional or (observation["load"][0] < MAX_LOAD and not observation["foreign_cpu_over_50"])
    )


def _cell(args, cell, ledger, functional):
    started = time.monotonic()
    ledger.current, ledger.stage = cell["id"], "local_diagnostic"
    rounds = 1 if functional else cell.get("rounds", 5)
    if rounds < 5 and not functional:
        raise ValueError("local diagnostics require at least five pairs")
    if len(cell["arms"]) != 2:
        raise ValueError("a diagnostic cell requires exactly two named arms")
    pairs = []
    seconds = {arm["name"]: [] for arm in cell["arms"]}
    try:
        for round_index in range(rounds):
            pair = _pair(args, cell, round_index, ledger, functional)
            for receipt in pair:
                seconds[receipt["arm"]].append(receipt["seconds"])
            by_arm = {r["arm"]: r for r in pair}
            names = [arm["name"] for arm in cell["arms"]]
            factor = by_arm[names[1]]["seconds"] / by_arm[names[0]]["seconds"]
            pairs.append(factor)
        ledger.result(
            cell,
            started,
            status="valid",
            local_diagnostic=True,
            publishable=False,
            pairs=pairs,
            paired_factor=statistics.median(pairs),
            ratio_min=min(pairs),
            ratio_max=max(pairs),
            ratio_summary=_summary(pairs),
            seconds_summary={name: _summary(values) for name, values in seconds.items()},
            quartile_method="inclusive linear interpolation",
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        ledger.result(
            cell,
            started,
            status="failed",
            local_diagnostic=True,
            publishable=False,
            error_type=type(error).__name__,
            reason=str(error),
        )


def _resolved(args, specification):
    context = {
        "candidate_python": sys.executable,
        "base_python": args.base_python,
        "reuse_work": str(args.reuse_work) if args.reuse_work else "",
    }
    for cell in specification["cells"]:
        for key in ("artifact", "input_file", "store", "ids_file"):
            if key in cell:
                cell[key] = cell[key].format_map(context)
        if "input_file" in cell:
            cell["input_sha256"] = sha256(Path(cell["input_file"]))
        for arm in cell["arms"]:
            arm["python"] = arm["python"].format_map(context)
            if "ids_file" in arm:
                arm["ids_file"] = arm["ids_file"].format_map(context)
    return specification


def run(args):
    if sys.platform != "darwin":
        raise RuntimeError("the local diagnostic mode is explicitly for macOS indications")
    specification = document(args.input)
    if specification.get("format") != "local-diagnostic-v1":
        raise ValueError("local-diagnostic-v1 specification required")
    if not args.bench_lock or not args.worker_name:
        raise ValueError("local diagnostics require --bench-lock and --worker-name")
    for path in (args.results, args.work):
        outside_checkout(path)
        path.mkdir(parents=True, exist_ok=False)
    specification = _resolved(args, specification)
    functional = args.diagnostic_rehearsal
    specification.update(functional_only=functional, local_diagnostic=True, publishable=False)
    atomic(
        args.results / "environment.json",
        {
            "machine": {
                "platform": platform.platform(),
                "cpu": subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip(),
            },
            "python": sys.version,
            "contract": specification,
            "telemetry": observations(),
            "measurement_source_sha256": {
                p.name: sha256(p) for p in sorted(ENTRY.parent.glob("*.py"))
            },
        },
    )
    ledger = Ledger(args.results, specification, time.time())
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(ledger, stop), daemon=True)
    thread.start()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: setattr(ledger, "stopping", True))
    try:
        for cell in specification["cells"]:
            if ledger.stopping:
                break
            _locked_cell(args, cell, ledger, functional)
    except BaseException as error:
        append(
            args.results / "fatal.jsonl", {"error_type": type(error).__name__, "reason": str(error)}
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
        stop.set()
        thread.join()
    return ledger.done()
