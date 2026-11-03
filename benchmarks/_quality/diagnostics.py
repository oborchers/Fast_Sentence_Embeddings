"""Fail-fast, full-output local quality gates through the central quality entry."""

from __future__ import annotations

import platform
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from _measurement import Ledger, atomic, document, heartbeat, sha256
from _measurement_config import MODEL_PINS, validate_cell
from _measurement_setup import outside_checkout

ENTRY = Path(__file__).resolve().parents[1] / "quality.py"


def _arm(args, cell, name, python):
    tag = f"{cell['id']}-{name}"
    job = {
        **cell,
        "work": str(args.work),
        "output": str(args.results / tag),
        "receipt": str(args.results / (tag + ".json")),
        "doc_ids": name == "candidate" and cell.get("candidate_doc_ids", False),
    }
    path = args.results / (tag + "-job.json")
    atomic(path, job)
    with (args.results / (tag + ".log")).open("x") as log:
        subprocess.run(
            [python, str(ENTRY), "--child-job", str(path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return document(Path(job["receipt"]))


def _same_bits(first, second):
    import numpy as np

    a = np.load(first["file"], mmap_mode="r", allow_pickle=False)
    b = np.load(second["file"], mmap_mode="r", allow_pickle=False)
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError("quality output shape or dtype differs")
    for begin in range(0, len(a), 4096):
        if not np.array_equal(
            a[begin : begin + 4096].view("uint8"), b[begin : begin + 4096].view("uint8")
        ):
            raise ValueError(f"quality output bits differ at block {begin}: {second['file']}")
    if (
        sha256(Path(first["file"])) != first["sha256"]
        or sha256(Path(second["file"])) != second["sha256"]
    ):
        raise ValueError("saved quality output changed during comparison")


def _compare(base, candidate):
    if base["encoder_fingerprint"] != candidate["encoder_fingerprint"]:
        raise ValueError("quality encoder fingerprints differ")
    compared = 0
    for mode, outputs in candidate["outputs"].items():
        reference = base["outputs"]["no-ids" if mode == "ids" else mode]
        if set(outputs) != set(reference):
            raise ValueError("quality output tensor sets differ")
        for name, output in outputs.items():
            _same_bits(reference[name], output)
            compared += 1
    if set(base["scores"]) != set(candidate["scores"]):
        raise ValueError("quality score splits differ")
    for split, scores in base["scores"].items():
        other = candidate["scores"][split]
        if scores["pairs"] != other["pairs"]:
            raise ValueError("quality score pair counts differ")
        for metric in ("pearson", "spearman"):
            if struct.pack("<d", scores[metric]) != struct.pack("<d", other[metric]):
                raise ValueError(f"STS-B {split} {metric} score bits differ")
    return compared


def _specification(args):
    spec = document(args.input)
    if spec.get("format") != "local-quality-diagnostic-v1":
        raise ValueError("local-quality-diagnostic-v1 specification required")
    identifiers = [c["id"] for c in spec["cells"]]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("nonempty unique quality cell identifiers required")
    context = {"reuse_work": str(args.reuse_work.resolve()) if args.reuse_work else ""}
    for cell in spec["cells"]:
        validate_cell(cell)
        if cell.get("model_revision", MODEL_PINS[cell["model"]]) != MODEL_PINS[cell["model"]]:
            raise ValueError("quality cell must use the shared model pin")
        cell["model_revision"] = MODEL_PINS[cell["model"]]
        for key in ("input", "artifact"):
            cell[key] = str(Path(cell[key].format_map(context)).resolve())
            digest = sha256(Path(cell[key]))
            if cell.get(key + "_sha256", digest) != digest:
                raise ValueError(f"quality {key} does not match its declared digest")
            cell[key + "_sha256"] = digest
    return spec


def _cell(args, cell, ledger):
    started = time.monotonic()
    ledger.current, ledger.stage = cell["id"], "full_output_quality_gate"
    base = _arm(args, cell, "base", args.base_python)
    candidate = _arm(args, cell, "candidate", sys.executable)
    expected = {"plain", "no-ids"} | ({"ids"} if cell.get("candidate_doc_ids") else set())
    if set(base["outputs"]) != {"plain", "no-ids"} or set(candidate["outputs"]) != expected:
        raise ValueError("quality output modes differ from the requested gate")
    compared = _compare(base, candidate)
    ledger.result(
        cell,
        started,
        status="valid",
        local_diagnostic=True,
        publishable=False,
        bit_identical=True,
        compared_tensors=compared,
        texts=base["texts"],
        scores=base["scores"],
        base_identity=base["identity"],
        candidate_identity=candidate["identity"],
    )


def run(args):
    if sys.platform != "darwin" or not args.base_python:
        raise ValueError("local quality diagnostics require macOS and --base-python")
    spec = _specification(args)
    for name in ("results", "work"):
        path = getattr(args, name).resolve()
        outside_checkout(path)
        path.mkdir(parents=True, exist_ok=False)
        setattr(args, name, path)
    spec.update(local_diagnostic=True, publishable=False)
    atomic(
        args.results / "environment.json",
        {
            "machine": {
                "platform": platform.platform(),
                "cpu": subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip(),
            },
            "contract": spec,
            "python": sys.version,
            "measurement_source_sha256": {
                str(p.relative_to(ENTRY.parent)): sha256(p)
                for p in sorted(ENTRY.parent.rglob("*.py"))
            },
        },
    )
    ledger = Ledger(args.results, spec, time.time())
    stop = threading.Event()
    watcher = threading.Thread(target=heartbeat, args=(ledger, stop), daemon=True)
    watcher.start()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: setattr(ledger, "stopping", True))
    try:
        for cell in spec["cells"]:
            if ledger.stopping:
                break
            try:
                _cell(args, cell, ledger)
            except Exception as error:
                ledger.result(
                    cell,
                    time.monotonic(),
                    status="failed",
                    local_diagnostic=True,
                    publishable=False,
                    reason=repr(error),
                )
                break  # A difference stops all further cells; never retry or relax it.
    finally:
        signal.signal(signal.SIGTERM, previous)
        stop.set()
        watcher.join()
    return ledger.done()
