"""Durable local results using the existing measurement collection contract."""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from _development_system import LABEL, machine, observations
from _measurement import Ledger, append, atomic, heartbeat, sha256
from _measurement_setup import outside_checkout


@contextmanager
def session(args, spec):
    for name in ("results", "work"):
        path = getattr(args, name).resolve()
        outside_checkout(path)
        path.mkdir(parents=True, exist_ok=False)
        setattr(args, name, path)
    spec = {**spec, **LABEL, "cells": [{**c, "estimated_seconds": 3600} for c in spec["cells"]]}
    atomic(
        args.results / "environment.json",
        {
            **LABEL,
            "machine": machine(),
            "contract": spec,
            "telemetry": observations(args.work),
            "measurement_source_sha256": {
                str(p.relative_to(Path(__file__).parent)): sha256(p)
                for p in sorted(Path(__file__).parent.rglob("*.py"))
                if "baseline" not in p.parts
            },
        },
    )
    ledger = Ledger(args.results, spec, time.time())
    stop = threading.Event()
    watcher = threading.Thread(target=heartbeat, args=(ledger, stop), daemon=True)
    previous = {
        signum: signal.signal(signum, lambda *_: setattr(ledger, "stopping", True))
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    watcher.start()
    try:
        yield ledger
    except Exception as error:
        append(args.results / "fatal.jsonl", {**LABEL, "error": repr(error)})
    finally:
        _missing(ledger)
        stop.set()
        watcher.join()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _missing(ledger):
    finished = {row["cell_id"] for row in ledger.finished}
    for cell in ledger.contract["cells"]:
        if cell["id"] not in finished:
            ledger.result(
                cell,
                time.monotonic(),
                **LABEL,
                status="not_started",
                reason="stop or earlier failure",
            )


def record_result(ledger, cell, started, **values):
    ledger.result({**cell, "estimated_seconds": 3600}, started, **LABEL, **values)
