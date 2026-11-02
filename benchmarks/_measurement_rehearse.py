"""Private built-in control and restart rehearsal, never publishable timing."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from _measurement import Ledger, atomic, document, sha256
from _measurement_config import CONFIG
from _measurement_setup import bootstrap


def control_case(root, name):
    root.mkdir()
    cells = [{"id": str(n), "estimated_seconds": 1} for n in range(2)]
    spec = {"format": "functional-control", "cells": cells,
            "machine": {"disk_gb": 1}, "functional_only": True,
            "billing": {"compute_usd_per_hour": 1, "disk_usd_per_gb_month": 0.1,
                        "month_hours": 720, "planning_minutes": 60, "planning_balance_usd": 2,
                        "collection_reserve_seconds": 1}}
    atomic(root / "environment.json", {"machine": {"scope": "functional-only"}})
    ledger = Ledger(root, spec, time.time() - (7200 if name == "budget" else 0))
    original = signal.signal(signal.SIGTERM, lambda *_: setattr(ledger, "stopping", True))
    try:
        for cell in cells:
            if name == "signal" and cell["id"] == "1":
                os.kill(os.getpid(), signal.SIGTERM)
            ledger.result(cell, time.monotonic(), status="valid" if ledger.admit(1) else "not_started")
        code = ledger.done()
    finally:
        signal.signal(signal.SIGTERM, original)
    expected = 0 if name == "normal" else 1
    if code != expected or document(root / "DONE")["exit_status"] != expected:
        raise ValueError("control rehearsal did not propagate terminal status")
    for file in ("cells.jsonl", "progress.json", "manifest.json", "DONE"):
        if not (root / file).is_file():
            raise ValueError("control rehearsal omitted a result file")
    return {"case": name, "exit_status": code, "cells": len(ledger.finished)}


def control_proof(root, entry):
    root.mkdir(parents=True, exist_ok=False)
    cases = [control_case(root / name, name) for name in ("normal", "signal", "budget")]
    script = root / "bootstrap.sh"
    script.write_text(bootstrap(entry, "0" * 40))
    env = {**os.environ, "MEASURE_CONTROL_PROBE": "1", "MEASURE_ROOT": str((root / "restart").resolve())}
    with (root / "bootstrap.log").open("w") as stream:
        first = subprocess.run(["bash", str(script)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        done = Path(env["MEASURE_ROOT"]) / (entry + "-results/DONE")
        before = sha256(done)
        second = subprocess.run(["bash", str(script)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if first.returncode != 0 or second.returncode != 77 or sha256(done) != before:
        raise ValueError("bootstrap restart did not preserve the existing completion")
    result = {"functional_only": True, "cases": cases, "restart_refused": True,
              "DONE_unchanged": True, "bootstrap_sha256": sha256(CONFIG / "bootstrap.sh"),
              "scope": "real result writer and signals; bootstrap restart gate with setup leaf disabled"}
    atomic(root / "proof.json", result)
    return result
