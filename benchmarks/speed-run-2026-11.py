"""Single entry for the frozen CPU comparison; no automatic machine lifecycle."""

from __future__ import annotations

import argparse
import json
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


def attempt(ledger, name, function, stage="setup", seconds=60):
    started, value = time.monotonic(), None
    record = dict(stage=name)
    try:
        if not ledger.admit(seconds, stage):
            raise RuntimeError(stage + " admission budget exhausted")
        value = function()
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=repr(error))
    record["seconds"] = time.monotonic() - started
    records.append(ledger.root / "preparation.jsonl", record)
    return value


def prepare(args, contract, ledger):
    work = args.work
    work.mkdir(parents=True, exist_ok=False)
    records.atomic(
        work / "owner.json",
        dict(script=records.sha256(Path(__file__)), results=str(args.results.resolve())),
    )
    models = {}
    for repo in contract["models"]:
        prepared = attempt(
            ledger,
            "model-" + repo.replace("/", "-"),
            lambda name=repo: engines.prepare_models(
                work, {**contract, "models": {name: contract["models"][name]}}
            ),
        )
        models.update(prepared or {})
    sources = attempt(
        ledger, "corpus", lambda: records.corpus_sources(contract, work / "datasets")
    ) or ([], [])
    rust = attempt(
        ledger, "rust", lambda: engines.build_rust(work / "native", HERE / "speed-rust.lock")
    )
    head = attempt(ledger, "head", lambda: engines.prepare_head(work, contract, models))
    return dict(
        models=models,
        paragraphs=sources[0],
        sentences=sources[1],
        rust=rust,
        head=head,
        inputs={},
        work=work,
        contract=contract,
        reference_python=args.reference_python,
        ledger=ledger,
    )


def input_file(context, cell):
    source = context["paragraphs"]
    repeated = cell["kind"] == "stream"
    if repeated:
        source = context["sentences"]
    if cell["kind"] == "classify":
        if context["head"] is None:
            raise ValueError("classification preparation failed")
        source = context["head"][0]
    key = (cell["kind"] == "classify", repeated, cell["texts"])
    if key not in context["inputs"]:
        token_cache = context.setdefault("token_lengths", {})
        if key[:2] not in token_cache:
            snapshot = context["models"]["minishlab/potion-base-8M"]["snapshot"]
            token_cache[key[:2]] = records.token_lengths(snapshot, source)
        path = context["work"] / ("input-" + "-".join(map(str, key)) + ".txt")
        identity = records.write_input(
            path, source, cell["texts"], 0, repeated=repeated, token_counts=token_cache[key[:2]]
        )
        context["inputs"][key] = (path, identity)
        records.append(
            context["ledger"].root / "input-identities.jsonl", dict(file=path.name, **identity)
        )
    return context["inputs"][key]


def labels(cell):
    if cell["kind"] == "classify":
        return ["fse-a", "fse-b", "unfolded", "model2vec-linear"]
    arms = ["fse-a", "fse-b"] + ([cell["reference"]] if cell.get("reference") else [])
    if cell.get("small_call_exception") and cell["workers"] == 16:
        arms += ["fse-reused", "rust-nested"]
    return arms


def job_for(context, cell, label, number, folder):
    path, identity = input_file(context, cell)
    model = context["models"][cell["model"]]
    backend = "fse" if label.startswith("fse-") else label
    if backend == "rust-nested":
        backend = "rust"
    if cell["kind"] == "classify" and backend == "fse":
        backend = "folded"
    job = dict(
        cell,
        label=label,
        round=number,
        backend=backend,
        persistent_pool=label == "fse-reused",
        nested_output=label == "rust-nested",
        input_file=str(path),
        input_sha256=identity["sha256"],
        snapshot=model["snapshot"],
        artifact=model["artifacts"][cell["dtype"]],
        vectors_file=str(folder / "vectors"),
        receipt=str(folder / "receipt.json"),
    )
    if backend == "rust":
        if not context["rust"]:
            raise ValueError("native build failed")
        job.update(context["rust"], rust_revision=context["rust"]["revision"])
    if cell["kind"] == "classify":
        job.update(context["head"][1])
    return job


def launch(context, cell, job, folder):
    root = context["ledger"].root
    filename = root / (cell["id"] + "-" + str(job["round"]) + "-" + job["label"] + ".job.json")
    records.atomic(filename, job)
    interpreter = context["reference_python"] if job["backend"] == "fastembed" else sys.executable
    with (root / (filename.stem + ".log")).open("x") as stream:
        process = subprocess.Popen(
            [interpreter, str(Path(__file__).resolve()), "--child-job", str(filename)],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            time.sleep(0.25)
    if process.returncode:
        raise RuntimeError(f"{job['backend']} exited {process.returncode}; see {filename.stem}.log")
    record = records.document(folder / "receipt.json")
    record.update(
        input_sha256=job["input_sha256"],
        model=job["model"],
        revision=context["models"][job["model"]]["revision"],
        adapter_sha256={
            p.name: records.sha256(p)
            for p in (
                Path(__file__).resolve(),
                HERE / "_speed_cells.py",
                HERE / "_speed_pipeline.py",
            )
        },
        input_identity=input_file(context, cell)[1],
    )
    tokens = record["input_identity"]["tokens_total"]
    record["mean_tokens"] = tokens / cell["texts"]
    record["tokens_per_second"] = [tokens / value for value in records.observations(record)]
    records.append(root / "operations.jsonl", dict(cell_id=cell["id"], **record))
    return record


def remove_output(context, path, record):
    if not path.resolve().is_relative_to(context["work"].resolve()) or path.is_symlink():
        raise ValueError("output ownership changed")
    if path.is_dir():
        actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        if actual != {r["path"] for r in record["files"]}:
            raise ValueError("stream output inventory changed")
        shutil.rmtree(path)
    else:
        path.unlink()


def stream_cell(context, cell):
    operations = []
    for number in range(cell["rounds"]):
        for label in labels(cell):
            folder = context["work"] / f"{cell['id']}-{number}-{label}"
            folder.mkdir()
            job = job_for(context, cell, label, number, folder)
            record = launch(context, cell, job, folder)
            if operations and record["vector_data_sha256"] != operations[0]["vector_data_sha256"]:
                raise ValueError("streamed A/A vector or quantization data differs")
            remove_output(context, Path(job["vectors_file"]), record)
            operations.append(record)
    return records.summarize(cell, operations)


def encode_cell(context, cell):

    path, _ = input_file(context, cell)
    dimension = 1024 if "static-retrieval" in cell["model"] else 256
    records.memory_admit(cell["texts"], dimension, path.stat().st_size)
    operations, reference, control = [], None, None
    for number in range(cell["rounds"]):
        order = labels(cell)
        order = order[number % len(order) :] + order[: number % len(order)]
        for label in order:
            folder = context["work"] / f"{cell['id']}-{number}-{label}"
            folder.mkdir()
            job = job_for(context, cell, label, number, folder)
            record = launch(context, cell, job, folder)
            output = records.output_array(job["vectors_file"], record)
            if reference is None:
                if label != "fse-a":
                    raise ValueError("first reference must be fse-a")
                reference, control = output, (Path(job["vectors_file"]), record)
            else:
                difference = records.check_output(
                    reference, output, label.startswith("fse-") or cell["kind"] == "classify"
                )
                records.append(
                    context["ledger"].root / "checks.jsonl",
                    dict(
                        cell_id=cell["id"],
                        label=label,
                        round=number,
                        max_abs_difference=difference,
                        rows=len(output),
                    ),
                )
                del output
                remove_output(context, Path(job["vectors_file"]), record)
            record["output_checked"] = True
            operations.append(record)
    del reference
    remove_output(context, *control)
    return records.summarize(cell, operations)


def execute_cell(context, cell):
    if cell.get("preparation_error"):
        raise ValueError(cell["preparation_error"])
    if cell["kind"] == "startup":
        if not context["rust"]:
            raise ValueError("native build failed")
        path = context["ledger"].root / (cell["id"] + ".startup.json")
        job = path.with_suffix(".job.json")
        records.atomic(job, dict(cell, binary=context["rust"]["binary"], receipt=str(path)))
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child-job", str(job)], check=True
        )
        return records.document(path)
    bytes_per_text = (
        300 if cell["kind"] == "stream" else 8192 if "static-retrieval" in cell["model"] else 2048
    )
    records.disk_admit(context["work"], cell["texts"] * bytes_per_text)
    return stream_cell(context, cell) if cell["kind"] == "stream" else encode_cell(context, cell)


def estimate(contract, cell):
    stage = next(r for r in contract["stages"] if r["name"] == cell["stage"])
    count = sum(r["stage"] == cell["stage"] for r in contract["cells"])
    return stage["estimate_minutes"] * 60 / count


def heartbeat(ledger, stop):
    while not stop.is_set():
        ledger.progress(
            getattr(ledger, "current", "preparation"), getattr(ledger, "stage", "setup")
        )
        stop.wait(15)


def quiet(context):
    ledger = context["ledger"]
    cell = dict(
        id="quiet",
        stage="quiet",
        kind="encode",
        workers=1,
        texts=1_000_000,
        model="minishlab/potion-base-8M",
        dtype="float32",
        rounds=3,
        warm_passes=3,
    )
    ledger.current, ledger.stage = cell["id"], cell["stage"]
    started = time.monotonic()
    result = encode_cell(context, cell)
    result["observation_seconds"] = time.monotonic() - started
    if result["observation_seconds"] < 120:
        result.update(status="invalid", reason="observation window shorter than 120 seconds")
    records.atomic(ledger.root / "quiet.json", result)
    return result


def sizing_job(context, repo, workers, backend, name, kind="encode"):
    ledger = context["ledger"]
    if not ledger.admit(60, "sizing"):
        raise RuntimeError("sizing admission budget exhausted")
    cell = dict(
        id=name,
        stage="sizing",
        kind=kind,
        workers=workers,
        texts=50_000,
        model=repo,
        dtype="float32",
        rounds=1,
    )
    folder = context["work"] / name
    folder.mkdir()
    job = job_for(context, cell, "fse-a" if backend in ("fse", "folded") else backend, 0, folder)
    ledger.current, ledger.stage = name, "sizing"
    record = launch(context, cell, job, folder)
    remove_output(context, Path(job["vectors_file"]), record)
    return 50_000 / min(record["first_seconds"], *record["warm_seconds"])


def apply_sizing(context, cell, rate, number):
    step = 100_000 if cell["kind"] == "classify" else 50_000
    requested = max(cell["texts"], int(3 * rate / step + 0.999999) * step)
    cell["texts"] = requested
    reference_rate = None
    if cell.get("reference"):
        reference_rate = attempt(
            context["ledger"],
            f"sizing-reference-{number}",
            lambda c=cell, n=number: sizing_job(
                context, c["model"], c["workers"], c["reference"], f"sizing-reference-{n}"
            ),
            stage="sizing",
        )
        if reference_rate is None:
            cell["preparation_error"] = "required reference sizing failed"
    calls = 1 + cell.get("warm_passes", 1)
    seconds_per_round = 2 * requested / rate
    if reference_rate:
        seconds_per_round += requested / reference_rate
    cell["estimated_seconds"] = max(
        estimate(context["contract"], cell), cell["rounds"] * calls * seconds_per_round
    )
    cell["sizing"] = dict(
        fse_texts_per_second=rate,
        reference_texts_per_second=reference_rate,
        target_fse_seconds=3,
        search_iterations=0,
    )


def size_once(context):
    contract, rates = context["contract"], {}
    for number, cell in enumerate(contract["cells"]):
        if cell["stage"] not in ("libraries", "classification"):
            continue
        key = (cell["model"], cell["workers"], cell["kind"])
        if key not in rates:
            rates[key] = attempt(
                context["ledger"],
                f"sizing-fse-{number}",
                lambda c=cell, n=number: sizing_job(
                    context, c["model"], c["workers"], "fse", f"sizing-fse-{n}", c["kind"]
                ),
                stage="sizing",
            )
        if rates[key] is None:
            cell["preparation_error"] = "required one-shot sizing failed"
            continue
        apply_sizing(context, cell, rates[key], number)
    records.atomic(context["ledger"].root / "sized-input.json", contract)


def scheduled_cell(context, cell, ledger, contract):
    started = time.monotonic()
    allowed = ledger.admit(ledger.estimate(cell, estimate(contract, cell)), cell["stage"])
    if not allowed:
        ledger.result(
            cell,
            "not_started",
            started,
            reason="stage or soft total admission budget exhausted",
        )
        return
    try:
        result = execute_cell(context, cell)
        ledger.result(cell, result.pop("status"), started, **result)
    except Exception as error:
        ledger.result(cell, "failed", started, reason=repr(error))


def validate_host(args):
    if sys.platform != "linux" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("The full measurement requires Linux and Python 3.13")
    if args.allocated_at is None:
        raise ValueError("--allocated-at must be the machine allocation Unix timestamp")


def run(args):
    validate_host(args)
    contract = records.document(args.input)
    args.results.mkdir(parents=True, exist_ok=False)
    ledger = records.Ledger(args.results, contract, args.allocated_at)
    records.metadata(args.results, contract)
    stop = threading.Event()
    watcher = threading.Thread(target=heartbeat, args=(ledger, stop), daemon=True)
    watcher.start()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: setattr(ledger, "stopping", True))
    try:
        context = prepare(args, contract, ledger)
        attempt(ledger, "quiet", lambda: quiet(context), stage="quiet", seconds=540)
        attempt(ledger, "sizing", lambda: size_once(context), stage="sizing", seconds=600)
        for cell in contract["cells"]:
            ledger.current, ledger.stage = cell["id"], cell["stage"]
            scheduled_cell(context, cell, ledger, contract)
    except Exception as error:
        records.append(args.results / "fatal.jsonl", dict(error=repr(error)))
        finished = {row["cell_id"] for row in ledger.finished}
        for cell in contract["cells"]:
            if cell["id"] not in finished:
                ledger.result(
                    cell,
                    "not_started",
                    time.monotonic(),
                    reason="run preparation failed: " + repr(error),
                )
    finally:
        stop.set()
        watcher.join()
        ledger.progress("DONE", "collection")
    return ledger.done()


def describe(contract):
    return {
        "variant": contract["variant"],
        "planned_cells": len(contract["cells"]),
        "planning_minutes": sum(s["budget_minutes"] for s in contract["stages"]),
        "hard_deadline": False,
        "cells": contract["cells"],
        "machine": contract["machine"],
        "python": sys.version.split()[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--child-job", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--setup", type=Path)
    parser.add_argument("--results", type=Path, default=Path("/speed-results"))
    parser.add_argument("--work", type=Path, default=Path("/speed-work"))
    parser.add_argument("--reference-python", default="/speed-env/reference/bin/python")
    parser.add_argument("--allocated-at", type=float)
    args = parser.parse_args()
    if args.setup:
        validate_host(args)
        engines.setup(args.setup, records.document(args.input), args.allocated_at)
        return 0
    if args.child_job:
        job = records.document(args.child_job)
        function = {"stream": engines.stream_child, "startup": records.startup_job}.get(
            job["kind"], engines.measure_child
        )
        function(job)
        return 0
    if args.run:
        return run(args)
    print(json.dumps(describe(records.document(args.input)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
