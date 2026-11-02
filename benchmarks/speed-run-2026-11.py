"""One fixed library-only CPU comparison; machine lifecycle belongs to the coordinator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import _speed_cells as records
import _speed_pipeline as engines

HERE = Path(__file__).resolve().parent
INPUT = HERE / "speed-input-2026-11.json"


class Ledger:
    def __init__(self, root, contract, allocated_at):
        self.root, self.contract, self.allocated_at = root, contract, allocated_at
        self.finished, self.stopping = [], False
        self.current, self.stage = "preparation", "setup"
        self.lock = threading.Lock()
        self.hourly = contract["billing"]["compute_usd_per_hour"] + (
            contract["machine"]["disk_gb"]
            * contract["billing"]["disk_usd_per_gb_month"]
            / contract["billing"]["month_hours"]
        )

    def admit(self, estimate):
        elapsed = time.time() - self.allocated_at
        ceiling = min(
            self.contract["billing"]["planning_minutes"] * 60,
            self.contract["billing"]["planning_balance_usd"] * 3600 / self.hourly,
        )
        return not self.stopping and elapsed + estimate + 600 <= ceiling

    def progress(self, terminal=False):
        with self.lock:
            remaining = [
                c
                for c in self.contract["cells"]
                if c["id"] not in {r["cell_id"] for r in self.finished}
            ]
            scale = (
                statistics.median(
                    [
                        r["duration_seconds"] / r["estimated_seconds"]
                        for r in self.finished
                        if r["status"] == "valid"
                    ]
                )
                if any(r["status"] == "valid" for r in self.finished)
                else 1
            )
            records.atomic(
                self.root / "progress.json",
                {
                    "stage": "collection" if terminal else self.stage,
                    "cell": "DONE" if terminal else self.current,
                    "finished": len(self.finished),
                    "planned": len(self.contract["cells"]),
                    "elapsed_seconds": time.time() - self.allocated_at,
                    "estimated_remaining_seconds": sum(c["estimated_seconds"] for c in remaining)
                    * scale,
                },
            )

    def result(self, cell, started, **result):
        row = {
            **cell,
            "cell_id": cell["id"],
            "run_id": self.contract["format"],
            "machine_environment_sha256": records.sha256(self.root / "environment.json"),
            "platform": platform.platform(),
            "python": sys.version,
            "completed_at_unix": time.time(),
            "date_utc": datetime.now(UTC).isoformat(),
            "machine": records.document(self.root / "environment.json")["machine"],
            "duration_seconds": time.monotonic() - started,
            "functional_only": self.contract.get("functional_only", False),
            **result,
        }
        records.append(self.root / "cells.jsonl", row)
        self.finished.append(row)
        self.progress()

    def done(self):
        self.progress(terminal=True)
        files = [
            {
                "path": str(p.relative_to(self.root)),
                "bytes": p.stat().st_size,
                "sha256": records.sha256(p),
            }
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", "DONE")
        ]
        records.atomic(self.root / "manifest.json", {"files": files})
        code = int(len(self.finished) != 8 or any(r["status"] != "valid" for r in self.finished))
        records.atomic(
            self.root / "DONE",
            {
                "exit_status": code,
                "completed_at": time.time(),
                "cells": len(self.finished),
                "manifest_sha256": records.sha256(self.root / "manifest.json"),
            },
        )
        return code


def prepare(args, contract, ledger):
    args.work.mkdir(parents=True, exist_ok=False)
    records.atomic(
        args.work / "owner.json",
        {"script_sha256": records.sha256(Path(__file__)), "results": str(args.results.resolve())},
    )
    models = engines.prepare_models(args.work, contract)
    paragraphs, unknown = records.corpus_sources(contract, args.work / "datasets")
    rust = engines.build_rust(args.work / "native", HERE / "speed-rust.lock")
    return {
        "work": args.work,
        "ledger": ledger,
        "models": models,
        "paragraphs": paragraphs,
        "unknown": unknown,
        "rust": rust,
        "reference_python": args.reference_python,
    }


def prepare_input(context, cell):
    source, mandatory = context["paragraphs"], []
    if cell["reference"] == "st":
        mandatory = context["unknown"]
    else:
        from fse import StaticModel

        model = StaticModel.from_pretrained(context["models"][cell["model"]]["snapshot"])
        if model.max_length != 512:
            raise ValueError("shared truncation requires the pinned 512-token rule")
        limit = model.max_length * model.median_token_length
        source = [text[:limit] for text in source]
    path = context["work"] / (cell["id"] + ".txt")
    identity = records.write_input(path, source, mandatory, cell["texts"], 0)
    identity["common_character_limit"] = None if mandatory else limit
    sample = path.with_suffix(".sample.json")
    records.atomic(sample, identity["sample_rows"])
    records.append(context["ledger"].root / "inputs.jsonl", dict(cell_id=cell["id"], **identity))
    return path, sample, identity


def launch(context, cell, label, number, path, sample, identity):
    root, work = context["ledger"].root, context["work"]
    folder = work / f"{cell['id']}-{number}-{label}"
    folder.mkdir()
    model = context["models"][cell["model"]]
    job = dict(
        cell,
        label=label,
        round=number,
        backend=label,
        input_file=str(path),
        input_sha256=identity["sha256"],
        sample_file=str(sample),
        snapshot=model["snapshot"],
        artifact=model["artifact"],
        vectors_file=str(folder / "sample"),
        receipt=str(folder / "receipt.json"),
    )
    if label == "rust":
        job.update(context["rust"], rust_revision=context["rust"]["revision"])
    spec = root / f"{cell['id']}-{number}-{label}.job.json"
    records.atomic(spec, job)
    python = context["reference_python"] if label == "fastembed" else sys.executable
    with spec.with_suffix(".log").open("x") as stream:
        child = subprocess.Popen(
            [python, str(Path(__file__).resolve()), "--child-job", str(spec)],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        while child.poll() is None:
            time.sleep(0.25)
    if child.returncode:
        raise RuntimeError(f"{label} failed with status {child.returncode}; see {spec.name}")
    record = records.document(folder / "receipt.json")
    record.update(
        label=label,
        round=number,
        input_sha256=identity["sha256"],
        model=cell["model"],
        revision=model["revision"],
        sample_file=str(folder / "sample"),
        adapter_sha256={
            p.name: records.sha256(p)
            for p in (Path(__file__), HERE / "_speed_cells.py", HERE / "_speed_pipeline.py")
        },
    )
    records.append(root / "operations.jsonl", dict(cell_id=cell["id"], **record))
    return record


def sample_array(record):
    import numpy as np

    path = record["sample_file"]
    if record.get("vector_format") == "raw":
        return np.memmap(path, dtype="<f4", mode="r", shape=tuple(record["shape"]))
    return np.load(path, mmap_mode="r", allow_pickle=False)


def cleanup(context, paths):
    for path in paths:
        if not path.resolve().is_relative_to(context["work"].resolve()) or path.is_symlink():
            raise ValueError("cleanup ownership changed")
        path.unlink()


def round_pair(context, cell, number, path, sample, identity):
    labels = ["fse", cell["reference"]]
    rows = [
        launch(context, cell, label, number, path, sample, identity)
        for label in (labels if number % 2 == 0 else labels[::-1])
    ]
    a = next(row for row in rows if row["label"] == "fse")
    b = next(row for row in rows if row["label"] != "fse")
    delta = records.check_output(sample_array(a), sample_array(b), False)
    records.append(
        context["ledger"].root / "checks.jsonl",
        {
            "cell_id": cell["id"],
            "round": number,
            "rows": len(identity["sample_rows"]),
            "mandatory_unknown_rows": identity["mandatory_prefix_rows"],
            "max_abs_difference": delta,
        },
    )
    cleanup(context, [Path(a["sample_file"]), Path(b["sample_file"])])
    return rows


def arm_summary(rows):
    seconds = [row["seconds"] for row in rows]
    return {
        "seconds": records.spread(seconds),
        "ordered_seconds": seconds,
        "identity": rows[0]["identity"],
        "versions": rows[0].get("versions", {}),
    }


def summarize_pair(cell, observations, identity):
    arms = {"fse": [], cell["reference"]: []}
    for row in observations:
        arms[row["label"]].append(row)
    for rows in arms.values():
        if [row["round"] for row in rows] != [0, 1, 2]:
            raise ValueError("missing or duplicate paired rounds")
        if len({json.dumps(r["identity"], sort_keys=True) for r in rows}) != 1:
            raise ValueError("executing backend changed between rounds")
    short = not cell.get("functional_only") and min(r["seconds"] for r in arms["fse"]) < 2
    ratios = [
        b["seconds"] / a["seconds"]
        for a, b in zip(arms["fse"], arms[cell["reference"]], strict=True)
    ]
    return {
        "status": "invalid" if short else "valid",
        "input_identity": identity,
        "reason": "fse call shorter than two seconds" if short else None,
        "sample_checked": True,
        "paired_speedup": records.spread(ratios),
        "arms": {name: arm_summary(rows) for name, rows in arms.items()},
    }


def pair(context, cell):
    path, sample, identity = prepare_input(context, cell)
    model_bytes = Path(context["models"][cell["model"]]["artifact"]).stat().st_size
    maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    if maximum == "max":
        raise ValueError("finite cgroup memory limit required")
    budget = records.memory_plan(cell, identity["file_bytes"], model_bytes, int(maximum))
    records.append(context["ledger"].root / "memory.jsonl", dict(cell_id=cell["id"], **budget))
    if not budget["admitted"]:
        raise MemoryError("fixed input exceeds the declared memory budget")
    observations = []
    for number in range(3):
        observations.extend(round_pair(context, cell, number, path, sample, identity))
    result = summarize_pair(cell, observations, identity)
    cleanup(context, [path, path.with_suffix(".json"), sample])
    return result


def quiet(context, contract):
    cell = dict(contract["quiet_cell"])
    path, sample, identity = prepare_input(context, cell)
    observations, started = [], time.monotonic()
    duration = 1 if contract.get("functional_only") else contract["rules"]["quiet_seconds"]
    while time.monotonic() - started < duration:
        row = launch(context, cell, "fse", len(observations), path, sample, identity)
        observations.append(row["seconds"])
        cleanup(context, [Path(row["sample_file"])])
    records.atomic(
        context["ledger"].root / "quiet.json",
        {
            "seconds": observations,
            "spread": records.spread(observations),
            "policy": "record only, no A/A pair rejection",
            "telemetry": records.telemetry(),
        },
    )
    cleanup(context, [path, path.with_suffix(".json"), sample])


def heartbeat(ledger, stop):
    while not stop.is_set():
        ledger.progress()
        stop.wait(15)


def write_environment(args, contract):
    records.atomic(
        args.results / "environment.json",
        {
            "contract": contract,
            "platform": platform.platform(),
            "python": sys.version,
            "telemetry": records.telemetry(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "machine": {
                "cpu": next(
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "affinity": sorted(os.sched_getaffinity(0)),
                "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
                "allocation_id": os.getenv("SPEED_ALLOCATION_ID"),
                "allocated_at_unix": args.allocated_at,
            },
        },
    )


def run(args):
    contract = records.document(args.input)
    if args.rehearse:
        contract["functional_only"] = True
        for cell in [*contract["cells"], contract["quiet_cell"]]:
            cell.update(texts=12_000, workers=min(cell["workers"], 2), functional_only=True)
        contract["billing"].update(planning_minutes=180, planning_balance_usd=10)
    args.results.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(args.results, contract, args.allocated_at)
    write_environment(args, contract)
    stop = threading.Event()
    watcher = threading.Thread(target=heartbeat, args=(ledger, stop), daemon=True)
    watcher.start()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: setattr(ledger, "stopping", True))
    context = None
    try:
        context = prepare(args, contract, ledger)
        ledger.stage, ledger.current = "quiet", "quiet"
        quiet(context, contract)
        ledger.stage = "libraries"
        for cell in contract["cells"]:
            ledger.current = cell["id"]
            started = time.monotonic()
            if not ledger.admit(cell["estimated_seconds"]):
                ledger.result(
                    cell, started, status="not_started", reason="soft budget or stop signal"
                )
                continue
            try:
                ledger.result(cell, started, **pair(context, cell))
            except Exception as error:
                ledger.result(cell, started, status="failed", reason=repr(error))
    except Exception as error:
        records.append(args.results / "fatal.jsonl", {"error": repr(error)})
    finally:
        finished = {r["cell_id"] for r in ledger.finished}
        for cell in contract["cells"]:
            if cell["id"] not in finished:
                ledger.result(
                    cell, time.monotonic(), status="not_started", reason="preparation failed"
                )
        stop.set()
        watcher.join()
    return ledger.done()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--child-job", type=Path)
    mode.add_argument("--setup", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--rehearse", action="store_true")
    parser.add_argument("--results", type=Path, default=Path("/speed-results"))
    parser.add_argument("--work", type=Path, default=Path("/speed-work"))
    parser.add_argument("--reference-python", default="/speed-env/reference/bin/python")
    parser.add_argument("--allocated-at", type=float)
    args = parser.parse_args()
    if args.child_job:
        engines.measure_child(records.document(args.child_job))
        return 0
    contract = records.document(args.input)
    if args.describe:
        print(json.dumps(contract, indent=2))
        return 0
    if sys.platform != "linux" or sys.version.split()[0] != contract["machine"]["python_patch"]:
        raise RuntimeError("pinned Linux Python required")
    if args.allocated_at is None:
        raise ValueError("allocation Unix timestamp required")
    if args.setup:
        engines.setup(args.setup, contract, args.allocated_at)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
