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


class LocalLedger(Ledger):
    """Keep every terminal/progress artifact explicitly separate from publication."""

    def progress(self, terminal=False):
        with self.lock:
            remaining = [
                c
                for c in self.contract["cells"]
                if c["id"] not in {r["cell_id"] for r in self.finished}
            ]
            atomic(
                self.root / "progress.json",
                {
                    **LABEL,
                    "stage": "collection" if terminal else self.stage,
                    "cell": "DONE" if terminal else self.current,
                    "finished": len(self.finished),
                    "planned": len(self.contract["cells"]),
                    "elapsed_seconds": time.time() - self.allocated_at,
                    "estimated_remaining_seconds": sum(c["estimated_seconds"] for c in remaining),
                },
            )

    def done(self):
        self.progress(terminal=True)
        files = [
            {"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", "DONE")
        ]
        atomic(self.root / "manifest.json", {**LABEL, "files": files})
        code = int(
            len(self.finished) != len(self.contract["cells"])
            or any(r["status"] != "valid" for r in self.finished)
            or (self.root / "fatal.jsonl").exists()
        )
        atomic(
            self.root / "DONE",
            {
                **LABEL,
                "exit_status": code,
                "completed_at": time.time(),
                "cells": len(self.finished),
                "manifest_sha256": sha256(self.root / "manifest.json"),
            },
        )
        return code


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
    ledger = LocalLedger(args.results, spec, time.time())
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
