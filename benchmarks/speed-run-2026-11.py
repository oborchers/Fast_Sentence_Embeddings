"""Paired encoding and streaming-write studies; machine lifecycle belongs to the coordinator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import _speed_cells as records
import _speed_pipeline as engines

HERE = Path(__file__).resolve().parent
INPUT = HERE / "speed-input-2026-11.json"


Ledger = records.Ledger


def require_budget(ledger, estimate):
    if not ledger.admit(estimate):
        raise RuntimeError("preparation admission budget or stop signal exhausted")


def prepare(args, contract, ledger):
    args.work.mkdir(parents=True, exist_ok=False)
    records.atomic(
        args.work / "owner.json",
        {"script_sha256": records.sha256(Path(__file__)), "results": str(args.results.resolve())},
    )
    setup_record = Path(sys.executable).parents[2] / "setup.json"
    if setup_record.exists():
        shutil.copyfile(setup_record, args.results / "setup.json")
    require_budget(ledger, 60)
    models = engines.prepare_models(args.work, contract, args.reuse_work)
    require_budget(ledger, 60)
    paragraphs, unknown = records.corpus_sources(
        contract, (args.reuse_work or args.work) / "datasets"
    )
    require_budget(ledger, 60)
    rust = (
        engines.build_rust(args.work / "native", HERE / "speed-rust.lock")
        if contract["cells"][0]["kind"] != "write"
        else None
    )
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
    if (
        not cell.get("functional_only")
        and cell.get("planned_input_bytes") is not None
        and identity["file_bytes"] != cell["planned_input_bytes"]
    ):
        raise ValueError("prepared input differs from the fixed production byte count")
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
    if cell["kind"] == "write":
        job["sample_texts_file"] = str(path.with_suffix(".sample-texts.json"))
        job["sample_texts_sha256"] = records.sha256(Path(job["sample_texts_file"]))
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


def memory_preflight(context, contract):
    maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    if maximum == "max":
        raise ValueError("finite cgroup memory limit required")
    target = contract["machine"]["ram_bytes"]
    limit = target if contract.get("functional_only") else min(target, int(maximum))
    rows = []
    for cell in contract["cells"]:
        production = dict(
            cell,
            texts=cell.get("production_texts", cell["texts"]),
            workers=cell.get("production_workers", cell["workers"]),
            functional_only=False,
        )
        model_bytes = Path(context["models"][cell["model"]]["artifact"]).stat().st_size
        plan = records.memory_plan(production, cell["planned_input_bytes"], model_bytes, limit)
        rows.append(dict(cell_id=cell["id"], **plan))
    records.atomic(
        context["ledger"].root / "preflight-memory.json",
        {"production_limit_bytes": target, "actual_cgroup_limit": int(maximum), "rows": rows},
    )
    if not all(row["admitted"] for row in rows):
        raise MemoryError("at least one fixed pair exceeds the preflight memory budget")


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
                "allocated_at_unix": float(
                    os.getenv("SPEED_MACHINE_ALLOCATED_AT", args.allocated_at)
                ),
                "budget_started_at_unix": args.allocated_at,
            },
        },
    )


def study_contract(args):
    contract = records.document(args.input)
    headline = args.study == "headline"
    contract["cells"] = [c for c in contract["cells"] if (c["kind"] == "write") == headline]
    if headline:
        if args.workers is None:
            raise ValueError("headline requires explicit --workers 16 or 32")
        contract["billing"].update(contract["headline_billing"])
        contract["format"] = "headline-2026-11-v1"
        contract["cells"][0]["workers"] = args.workers
    if args.rehearse:
        contract["rehearsal_limits"] = records.rehearsal_limits()
        contract["functional_only"] = True
        for cell in [*contract["cells"], contract["quiet_cell"]]:
            cell["production_workers"], cell["production_texts"] = cell["workers"], cell["texts"]
            cell.update(
                texts=1_000_000 if headline else 12_000,
                workers=min(cell["workers"], 2),
                functional_only=True,
            )
        contract["billing"].update(planning_minutes=180, planning_balance_usd=10)
    return contract


def run(args):
    contract = study_contract(args)
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
        require_budget(ledger, 60)
        context = prepare(args, contract, ledger)
        action = run_headline if args.study == "headline" else run_libraries
        action(context, contract)
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


def run_libraries(context, contract):
    ledger = context["ledger"]
    memory_preflight(context, contract)
    ledger.stage, ledger.current = "quiet", "quiet"
    require_budget(
        ledger, 1 if contract.get("functional_only") else contract["rules"]["quiet_seconds"]
    )
    quiet(context, contract)
    ledger.stage = "libraries"
    for cell in contract["cells"]:
        ledger.current = cell["id"]
        started = time.monotonic()
        if not ledger.admit(cell["estimated_seconds"]):
            ledger.result(cell, started, status="not_started", reason="soft budget or stop signal")
            continue
        try:
            ledger.result(cell, started, **pair(context, cell))
        except Exception as error:
            ledger.result(cell, started, status="failed", reason=repr(error))


def run_headline(context, contract):

    ledger, work = context["ledger"], context["work"]
    cell = contract["cells"][0]
    ledger.stage, ledger.current = "headline", cell["id"]
    started = time.monotonic()
    if not ledger.admit(cell["estimated_seconds"]):
        ledger.result(cell, started, status="not_started", reason="headline admission budget")
        return
    source = context["paragraphs"]
    records.stream_budget(work, cell, source)
    tokens = engines.headline_tokens(context["models"][cell["model"]]["snapshot"], source)
    path = work / "headline-input.txt"
    identity = records.write_input(path, source, [], cell["texts"], 0, tokens)
    records.append(ledger.root / "inputs.jsonl", dict(cell_id=cell["id"], **identity))
    sample = path.with_suffix(".sample.json")
    records.atomic(sample, identity["sample_rows"])
    rows = []
    for number in range(cell["rounds"]):
        if not ledger.admit(120):
            break
        row = launch(context, cell, "fse", number, path, sample, identity)
        output = Path(row["sample_file"])
        actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
        if output.is_symlink() or not output.resolve().is_relative_to(work.resolve()):
            raise ValueError("headline cleanup ownership changed")
        if actual != {r["path"] for r in row["files"]}:
            raise ValueError("headline output inventory changed before cleanup")
        shutil.rmtree(output)
        records.append(
            ledger.root / "cleanup.jsonl",
            dict(
                round=number,
                completed_at=time.time(),
                disk_after=shutil.disk_usage(work)._asdict(),
            ),
        )
        rows.append(row)
    complete = len(rows) == cell["rounds"]
    ledger.result(
        cell,
        started,
        status="valid" if complete else "partial",
        input_identity=identity,
        rounds_completed=len(rows),
        sample_checked=bool(rows),
        seconds=records.spread([r["seconds"] for r in rows]) if rows else None,
        texts_per_second=records.spread([cell["texts"] / r["seconds"] for r in rows])
        if rows
        else None,
        mean_tokens_per_text=identity["mean_tokenizer_ids_per_text"],
        tokens_per_second=records.spread(
            [cell["texts"] * identity["mean_tokenizer_ids_per_text"] / r["seconds"] for r in rows]
        )
        if rows
        else None,
        reason=None if complete else "stop signal or budget before next round",
    )
    cleanup(
        context, [path, path.with_suffix(".json"), sample, path.with_suffix(".sample-texts.json")]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--study", choices=("libraries", "headline"), default="libraries")
    parser.add_argument("--workers", type=int, choices=(16, 32))
    parser.add_argument("--reuse-work", type=Path)
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
        if os.getenv("SPEED_REHEARSAL") == "1":
            records.rehearsal_limits()
        engines.setup(args.setup, contract, args.allocated_at)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
