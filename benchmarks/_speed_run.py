"""Paired encoding and streaming-write studies; machine lifecycle belongs to the coordinator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _measurement as records
import _speed_cells as inputs
import _speed_pipeline as engines
import _speed_protocol as protocols
import _speed_write as writing
from _harness import records as durable
from _harness.ledger import Ledger, watching
from _measurement import write_environment
from _measurement_config import CONFIG
from _measurement_config import contract as load_contract

ENTRY = Path(__file__).resolve().with_name("speed.py")

HERE = Path(__file__).resolve().parent


def require_budget(ledger, estimate):
    if not ledger.admit(estimate):
        raise RuntimeError("preparation admission budget or stop signal exhausted")


def prepare(args, contract, ledger):
    args.work.mkdir(parents=True, exist_ok=False)
    durable.sync_directory(args.work.parent)
    durable.atomic(
        args.work / "owner.json",
        {"script_sha256": durable.sha256(ENTRY), "results": str(args.results.resolve())},
    )
    setup_record = Path(sys.executable).parents[2] / "setup.json"
    if setup_record.exists():
        shutil.copyfile(setup_record, args.results / "setup.json")
    require_budget(ledger, 60)
    models = engines.prepare_models(args.work, contract, args.reuse_work)
    require_budget(ledger, 60)
    sentences, unknown, paragraphs = inputs.corpus_sources(
        contract, (args.reuse_work or args.work) / "datasets"
    )
    require_budget(ledger, 60)
    rust = (
        engines.build_rust(args.work / "native", CONFIG / "rust.lock")
        if any(c.get("reference") == "rust" for c in contract["cells"])
        else None
    )
    return {
        "work": args.work,
        "ledger": ledger,
        "models": models,
        "paragraphs": sentences,
        "paragraph_texts": paragraphs,
        "unknown": unknown,
        "rust": rust,
        "reference_python": args.reference_python,
        "base_python": args.base_python,
    }


def prepare_input(context, cell):
    paragraphs = cell.get("corpus", "sentences") == "paragraphs"
    source, mandatory = context["paragraph_texts" if paragraphs else "paragraphs"], []
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
    unit = "paragraphs" if paragraphs else "sentences"
    identity = inputs.write_input(path, source, mandatory, cell["texts"], 0, unit=unit)
    if (
        not cell.get("functional_only")
        and cell.get("planned_input_bytes") is not None
        and identity["file_bytes"] != cell["planned_input_bytes"]
    ):
        raise ValueError("prepared input differs from the fixed production byte count")
    identity["common_character_limit"] = None if mandatory else limit
    sample = path.with_suffix(".sample.json")
    durable.atomic(sample, identity["sample_rows"])
    durable.append(context["ledger"].root / "inputs.jsonl", dict(cell_id=cell["id"], **identity))
    return path, sample, identity


def launch(context, cell, label, number, path, sample, identity):
    root, work = context["ledger"].root, context["work"]
    folder = work / f"{cell['id']}-{number}-{label}"
    folder.mkdir()
    durable.sync_directory(work)
    model = context["models"][cell["model"]]
    job = dict(
        cell,
        label=label,
        round=number,
        backend="fse" if label.startswith("fse-") else label,
        input_file=str(path),
        input_sha256=identity["sha256"],
        sample_file=str(sample),
        snapshot=model["snapshot"],
        artifact=model.get("artifacts", {}).get(
            cell.get("table_dtype", cell["dtype"]), model["artifact"]
        ),
        vectors_file=str(folder / "sample"),
        receipt=str(folder / "receipt.json"),
    )
    if label == "rust":
        job.update(context["rust"], rust_revision=context["rust"]["revision"])
    if cell["kind"] == "write":
        job["sample_texts_file"] = str(path.with_suffix(".sample-texts.json"))
        job["sample_texts_sha256"] = durable.sha256(Path(job["sample_texts_file"]))
    spec = root / f"{cell['id']}-{number}-{label}.job.json"
    durable.atomic(spec, job)
    python = engines.interpreter(job["backend"], context["reference_python"])
    if label == "fse-base":
        if not context["base_python"]:
            raise ValueError("fse-base requires --base-python")
        python = context["base_python"]
    with spec.with_suffix(".log").open("x") as stream:
        child = subprocess.Popen(
            [python, str(ENTRY), "--child-job", str(spec)],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        while child.poll() is None:
            time.sleep(0.25)
    if child.returncode:
        raise RuntimeError(f"{label} failed with status {child.returncode}; see {spec.name}")
    record = durable.document(folder / "receipt.json")
    record.update(
        label=label,
        round=number,
        input_sha256=identity["sha256"],
        model=cell["model"],
        revision=model["revision"],
        sample_file=str(folder / "sample"),
        adapter_sha256={p.name: durable.sha256(p) for p in [*sorted(HERE.glob("_*.py")), ENTRY]},
    )
    durable.append(root / "operations.jsonl", dict(cell_id=cell["id"], **record))
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
    delta = inputs.check_output(sample_array(a), sample_array(b), protocols.protocol(cell)["exact"])
    durable.append(
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
    validate_pair_arms(arms)
    short, aa_upper, noisy = pair_validity(cell, observations, arms)
    ratios = [
        b["seconds"] / a["seconds"]
        for a, b in zip(arms["fse"], arms[cell["reference"]], strict=True)
    ]

    return {
        "protocol": protocols.protocol(cell)["name"],
        "status": "invalid" if short or noisy else "valid",
        "aa": {"status": "measured", "symmetric_upper": aa_upper}
        if aa_upper is not None
        else {"status": "not_run"},
        "input_identity": identity,
        "reason": "call shorter than two seconds"
        if short
        else ("A/A exceeds 10 percent" if noisy else None),
        "sample_checked": True,
        "paired_speedup": records.spread(ratios),
        "arms": {name: arm_summary(rows) for name, rows in arms.items()},
    }


def memory_preflight(context, contract):
    maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    if maximum == "max":
        raise ValueError("finite cgroup memory limit required")
    target = contract["machine"]["ram_bytes"]
    declared = contract.get("functional_only") or target <= int(maximum)
    limit = (target, "declared") if declared else (int(maximum), "cgroup")
    rows = []
    for cell in contract["cells"]:
        production = dict(
            cell,
            texts=cell.get("production_texts", cell["texts"]),
            workers=cell.get("production_workers", cell["workers"]),
            functional_only=False,
        )
        if cell["kind"] == "write":
            continue
        model_bytes = Path(context["models"][cell["model"]]["artifact"]).stat().st_size
        plan = inputs.memory_plan(production, cell["planned_input_bytes"], model_bytes, limit)
        rows.append(dict(cell_id=cell["id"], **plan))
    durable.atomic(
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
    kind = "functional" if cell.get("functional_only") else "production"
    budget = inputs.memory_plan(
        cell, identity["file_bytes"], model_bytes, (int(maximum), "cgroup"), kind
    )
    durable.append(context["ledger"].root / "memory.jsonl", dict(cell_id=cell["id"], **budget))
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
    durable.atomic(
        context["ledger"].root / "quiet.json",
        {
            "seconds": observations,
            "spread": records.spread(observations),
            "policy": "record only, no A/A pair rejection",
            "telemetry": records.telemetry(),
        },
    )
    cleanup(context, [path, path.with_suffix(".json"), sample])


def study_contract(args):
    contract = load_contract(args.input)
    headline = args.study == "headline"
    if args.study != "all":
        contract["cells"] = [c for c in contract["cells"] if c["stage"] == args.study]
    if not contract["cells"]:
        raise ValueError("study has no matching cells")
    protocols.validate(contract["cells"])
    protocols.require(contract["cells"], args.base_python)
    if headline:
        if args.workers is None:
            raise ValueError("headline requires explicit --workers 16 or 32")
        contract["billing"].update(contract["headline_billing"])
        contract["format"] = "headline-study-v1"
        contract["cells"][0]["workers"] = args.workers
    if args.rehearse:
        contract["rehearsal_limits"] = records.rehearsal_limits()
        contract["functional_only"] = True
        for cell in [*contract["cells"], contract["quiet_cell"]]:
            cell["production_workers"], cell["production_texts"] = cell["workers"], cell["texts"]
            cell.update(
                texts=20_000 if cell["kind"] == "write" else 12_000,
                workers=min(cell["workers"], 2),
                functional_only=True,
            )
        contract["billing"].update(planning_minutes=180, planning_balance_usd=10)
    return contract


def run(args):
    contract = study_contract(args)
    args.results.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(args.results, contract, args.allocated_at)
    os.environ["MEASURE_WORK"] = str(args.work)
    write_environment(args, contract)
    context = None
    with watching(ledger, restore=False):
        try:
            require_budget(ledger, 60)
            context = prepare(args, contract, ledger)
            run_cells(context, contract)
        except Exception as error:
            durable.append(args.results / "fatal.jsonl", {"error": repr(error)})
        finally:
            ledger.fill_missing("preparation failed")
    records.finalize_rehearsal(ledger, "speed")
    return ledger.done()


def run_cells(context, contract):
    ledger = context["ledger"]
    memory_preflight(context, contract)
    if any(c["kind"] != "write" for c in contract["cells"]):
        ledger.stage, ledger.current = "quiet", "quiet"
        require_budget(
            ledger, 1 if contract.get("functional_only") else contract["rules"]["quiet_seconds"]
        )
        quiet(context, contract)
    for cell in contract["cells"]:
        ledger.current, ledger.stage = cell["id"], cell["stage"]
        started = time.monotonic()
        if not ledger.admit(cell["estimated_seconds"]):
            ledger.result(cell, started, status="not_started", reason="soft budget or stop signal")
            continue
        try:
            if cell["kind"] == "write":
                run_headline(context, {**contract, "cells": [cell]})
            else:
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
    inputs.stream_budget(work, cell, source)
    tokens = writing.headline_tokens(context["models"][cell["model"]]["snapshot"], source)
    path = work / "headline-input.txt"
    identity = inputs.write_input(path, source, [], cell["texts"], 0, tokens, ledger.admit)
    durable.append(ledger.root / "inputs.jsonl", dict(cell_id=cell["id"], **identity))
    sample = path.with_suffix(".sample.json")
    durable.atomic(sample, identity["sample_rows"])
    rows, error = inputs.headline_rounds(context, cell, path, sample, identity, launch, cleanup)
    complete = len(rows) == cell["rounds"] and error is None
    ledger.result(
        cell,
        started,
        protocol=protocols.protocol(cell)["name"],
        status="failed" if error else ("valid" if complete else "partial"),
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
        reason=error or (None if complete else "stop signal or budget before next round"),
    )


def pair_validity(cell, observations, arms):
    measured_fse = observations if cell["reference"].startswith("fse-") else arms["fse"]
    short = not cell.get("functional_only") and min(r["seconds"] for r in measured_fse) < 2
    if not protocols.protocol(cell)["control"]:
        return short, None, False
    a = records.spread([r["seconds"] for r in arms["fse"]])
    b = records.spread([r["seconds"] for r in arms[cell["reference"]]])
    upper = max(a["max"] / b["min"], b["max"] / a["min"])
    return short, upper, upper > 1.10 and not cell.get("functional_only")


def validate_pair_arms(arms):
    for rows in arms.values():
        if [row["round"] for row in rows] != [0, 1, 2]:
            raise ValueError("missing or duplicate paired rounds")
        if len({json.dumps(r["identity"], sort_keys=True) for r in rows}) != 1:
            raise ValueError("executing backend changed between rounds")
