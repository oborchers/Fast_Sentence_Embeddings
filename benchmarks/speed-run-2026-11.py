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

from _speed_cells import (
    Ledger,
    append,
    atomic,
    check_output,
    corpus_sources,
    describe,
    document,
    memory_admit,
    metadata,
    output_array,
    sha256,
    startup_job,
    summarize,
    write_input,
)
from _speed_pipeline import (
    build_rust,
    measure_child,
    prepare_head,
    prepare_models,
    setup,
    stream_child,
)

HERE = Path(__file__).resolve().parent
INPUT = HERE / "speed-input-2026-11.json"


def attempt(root, name, function):
    started = time.monotonic()
    try:
        value = function()
        append(
            root / "preparation.jsonl",
            dict(stage=name, status="complete", seconds=time.monotonic() - started),
        )
        return value
    except Exception as error:
        append(
            root / "preparation.jsonl",
            dict(
                stage=name, status="failed", seconds=time.monotonic() - started, error=repr(error)
            ),
        )
        return None


def prepare(args, contract, ledger):
    work = args.work
    work.mkdir(parents=True, exist_ok=False)
    atomic(
        work / "owner.json",
        dict(script=sha256(Path(__file__)), results=str(args.results.resolve())),
    )
    models = {}
    for repo in contract["models"]:
        prepared = attempt(
            args.results,
            "model-" + repo.replace("/", "-"),
            lambda name=repo: prepare_models(
                work, {**contract, "models": {name: contract["models"][name]}}
            ),
        )
        models.update(prepared or {})
    sources = attempt(
        args.results, "corpus", lambda: corpus_sources(contract, work / "datasets")
    ) or ([], [])
    rust = attempt(
        args.results, "rust", lambda: build_rust(work / "native", HERE / "speed-rust.lock")
    )
    head = attempt(args.results, "head", lambda: prepare_head(work, contract, models))
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
        path = context["work"] / ("input-" + "-".join(map(str, key)) + ".txt")
        identity = write_input(path, source, cell["texts"], 0, repeated=repeated)
        context["inputs"][key] = (path, identity)
        append(context["ledger"].root / "input-identities.jsonl", dict(file=path.name, **identity))
    return context["inputs"][key]


def labels(cell):
    if cell["kind"] == "classify":
        return ["fse-a", "fse-b", "unfolded", "model2vec-linear"]
    return ["fse-a", "fse-b"] + ([cell["reference"]] if cell.get("reference") else [])


def job_for(context, cell, label, number, folder):
    path, identity = input_file(context, cell)
    model = context["models"][cell["model"]]
    backend = "fse" if label.startswith("fse-") else label
    if cell["kind"] == "classify" and backend == "fse":
        backend = "folded"
    job = dict(
        cell,
        label=label,
        round=number,
        backend=backend,
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
    atomic(filename, job)
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
    record = document(folder / "receipt.json")
    record.update(
        input_sha256=job["input_sha256"],
        model=job["model"],
        revision=context["models"][job["model"]]["revision"],
        adapter_sha256={
            p.name: sha256(p)
            for p in (
                Path(__file__).resolve(),
                HERE / "_speed_cells.py",
                HERE / "_speed_pipeline.py",
            )
        },
        input_identity=input_file(context, cell)[1],
    )
    append(root / "operations.jsonl", dict(cell_id=cell["id"], **record))
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
            remove_output(context, Path(job["vectors_file"]), record)
            operations.append(record)
    return summarize(cell, operations)


def encode_cell(context, cell):

    path, _ = input_file(context, cell)
    dimension = 1024 if "static-retrieval" in cell["model"] else 256
    memory_admit(cell["texts"], dimension, path.stat().st_size)
    operations, reference, control = [], None, None
    for number in range(cell["rounds"]):
        order = labels(cell)
        order = order[number % len(order) :] + order[: number % len(order)]
        for label in order:
            folder = context["work"] / f"{cell['id']}-{number}-{label}"
            folder.mkdir()
            job = job_for(context, cell, label, number, folder)
            record = launch(context, cell, job, folder)
            output = output_array(job["vectors_file"], record)
            if reference is None:
                if label != "fse-a":
                    raise ValueError("first reference must be fse-a")
                reference, control = output, (Path(job["vectors_file"]), record)
            else:
                difference = check_output(
                    reference, output, label.startswith("fse-") or cell["kind"] == "classify"
                )
                append(
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
    return summarize(cell, operations)


def execute_cell(context, cell):
    if cell.get("preparation_error"):
        raise ValueError(cell["preparation_error"])
    if cell["kind"] == "startup":
        if not context["rust"]:
            raise ValueError("native build failed")
        path = context["ledger"].root / (cell["id"] + ".startup.json")
        startup_job(dict(cell, binary=context["rust"]["binary"], receipt=str(path)))
        return document(path)
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
    atomic(ledger.root / "quiet.json", result)
    return result


def sizing_job(context, repo, workers, backend, name, kind="encode"):
    ledger = context["ledger"]
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
            context["ledger"].root,
            f"sizing-reference-{number}",
            lambda c=cell, n=number: sizing_job(
                context, c["model"], c["workers"], c["reference"], f"sizing-reference-{n}"
            ),
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
                context["ledger"].root,
                f"sizing-fse-{number}",
                lambda c=cell, n=number: sizing_job(
                    context, c["model"], c["workers"], "fse", f"sizing-fse-{n}", c["kind"]
                ),
            )
        if rates[key] is None:
            cell["preparation_error"] = "required one-shot sizing failed"
            continue
        apply_sizing(context, cell, rates[key], number)
    atomic(context["ledger"].root / "sized-input.json", contract)


def scheduled_cell(context, cell, ledger, deadline, interrupted, contract):
    started = time.monotonic()
    allowed = (
        not interrupted.is_set()
        and started < deadline
        and ledger.admit(ledger.estimate(cell, estimate(contract, cell)))
    )
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


def run(args):
    contract = document(args.input)
    args.results.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(args.results, contract, args.allocated_at)
    metadata(args.results, contract)
    stop = threading.Event()
    watcher = threading.Thread(target=heartbeat, args=(ledger, stop), daemon=True)
    watcher.start()
    interrupted = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: interrupted.set())
    try:
        context = prepare(args, contract, ledger)
        attempt(args.results, "quiet", lambda: quiet(context))
        attempt(args.results, "sizing", lambda: size_once(context))
        deadlines = {}
        for cell in contract["cells"]:
            ledger.current, ledger.stage = cell["id"], cell["stage"]
            budget = next(
                s["budget_minutes"] * 60 for s in contract["stages"] if s["name"] == cell["stage"]
            )
            deadlines.setdefault(cell["stage"], time.monotonic() + budget)
            scheduled_cell(context, cell, ledger, deadlines[cell["stage"]], interrupted, contract)
    except Exception as error:
        append(args.results / "fatal.jsonl", dict(error=repr(error)))
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
    parser.add_argument("--reference-python", default="/speed-reference/bin/python")
    parser.add_argument("--allocated-at", type=float, default=time.time())
    args = parser.parse_args()
    if args.setup:
        setup(args.setup, document(args.input))
        return 0
    if args.child_job:
        job = document(args.child_job)
        stream_child(job) if job["kind"] == "stream" else measure_child(job)
        return 0
    if args.run:
        return run(args)
    print(json.dumps(describe(document(args.input)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
