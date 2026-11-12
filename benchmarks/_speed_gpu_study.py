"""Sealed per-model CUDA probe verdicts, using the existing allocation step receipt."""

from __future__ import annotations

import time
from pathlib import Path

from _harness.prerequisites import receipt, runner_identity
from _harness.records import atomic, document, read_rows, sha256
from _measurement_config import CONFIG


def identity(context, contract):
    root = context["allocation_root"]
    return {
        "allocation_root": str(root.resolve()) if root is not None else None,
        "allocated_at": context["allocated_at"],
        "cells": [
            {k: v for k, v in cell.items() if k != "gpu_phase"} for cell in contract["cells"]
        ],
        "models": {cell["model"]: contract["models"][cell["model"]] for cell in contract["cells"]},
        "lock_sha256": sha256(CONFIG / "speed-gpu.lock.txt"),
        "configuration_sha256": sha256(CONFIG / "speed.json"),
    }


def probe_verdicts(context, contract, expected_identity=None):
    """Require the existing allocation receipt, a complete seal and identical controls."""
    root = context["allocation_root"]
    if root is None:
        raise ValueError("GPU measurement requires --allocation-root and its sealed gpu-probe")
    results = root / "gpu-probe-results"
    rows = [
        r for r in read_rows(root / "allocation-results", "steps.jsonl") if r["step"] == "gpu-probe"
    ]
    if len(rows) != 1 or rows[0]["status"] != "finished" or rows[0]["exit_status"] != 0:
        raise ValueError("GPU probe has no successful allocation step")
    evidence = rows[0]["evidence"]
    origin = evidence["identity"]
    if origin["root"] != str(root.resolve()) or origin["allocated_at"] != context["allocated_at"]:
        raise ValueError("GPU probe belongs to another allocation")
    if origin["runner"] != runner_identity():
        raise ValueError("GPU probe runner revision or sources differ")
    if any(sha256(Path(p)) != digest for p, digest in origin["files"].items()):
        raise ValueError("GPU probe lock or configuration changed")
    if receipt(results, origin) != evidence:
        raise ValueError("GPU probe seal differs from the allocation receipt")
    expected = identity(context, contract) if expected_identity is None else expected_identity
    return saved_verdicts(results, expected)


def saved_verdicts(results, expected):
    try:
        saved = document(results / "gpu-verdicts.json")
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    if saved.get("identity") != expected:
        raise ValueError("GPU probe model revisions, allocation, lock or cell controls differ")
    return saved.get("verdicts", {}) if isinstance(saved.get("verdicts"), dict) else {}


def verdict(verdicts, cell):
    row = verdicts.get(cell["id"])
    if not isinstance(row, dict) or row.get("verdict") not in ("passed", "failed"):
        return {"verdict": "failed", "reason": "GPU probe verdict is missing or unreadable"}
    if row["verdict"] == "failed" and not isinstance(row.get("reason"), str):
        return {"verdict": "failed", "reason": "GPU probe failure reason is unreadable"}
    return row


def probe(context, cell):
    from _speed_run import cleanup, launch, prepare_input
    from _speed_transformer import admit

    path, sample, input_identity = prepare_input(context, cell)
    admit(context, cell, input_identity)
    row = launch(context, cell, "st", 0, path, sample, input_identity)
    if row.get("gpu_verdict") not in ("passed", "failed"):
        raise ValueError("GPU child supplied no model verdict")
    if row["gpu_verdict"] == "passed" and not row.get("gpu_reference_check"):
        raise ValueError("GPU child supplied no reference check")
    paths = [path, path.with_suffix(".json"), sample]
    if Path(row["sample_file"]).exists():
        paths.append(Path(row["sample_file"]))
    cleanup(context, paths)
    return {
        "verdict": row["gpu_verdict"],
        "reason": row.get("reason"),
        "input_identity": input_identity,
        "reference_check": row.get("gpu_reference_check"),
    }


def run(context, contract):
    from _speed_transformer import run as measure

    ledger = context["ledger"]
    phase = contract["cells"][0]["gpu_phase"]
    verdicts = {} if phase == "probe" else probe_verdicts(context, contract)
    checked = 0
    for cell in contract["cells"]:
        ledger.current, ledger.stage = cell["id"], "gpu"
        started = time.monotonic()
        allowed = verdict(verdicts, cell)
        if phase == "measure" and allowed["verdict"] != "passed":
            ledger.result(cell, started, status="not_started", reason=allowed["reason"])
            continue
        if not ledger.admit(cell["estimated_seconds"]):
            ledger.result(cell, started, status="not_started", reason="budget or stop signal")
            continue
        if phase == "probe":
            verdicts[cell["id"]] = probe(context, cell)
            checked += 1
            atomic(
                ledger.root / "gpu-verdicts.json",
                {"identity": identity(context, contract), "verdicts": verdicts},
            )
            ledger.result(
                cell,
                started,
                protocol="transformer@1",
                status="valid",
                speed_evidence=False,
                **verdicts[cell["id"]],
            )
        else:
            try:
                ledger.result(cell, started, **measure(context, cell))
            except Exception as error:
                ledger.result(cell, started, status="failed", reason=repr(error))
    if phase == "probe" and not checked:
        raise ValueError("GPU probe checked no models")
