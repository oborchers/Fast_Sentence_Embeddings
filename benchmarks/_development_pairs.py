"""Whole-call paired blocks, sharing one cell lock and one frozen workload."""

from __future__ import annotations

import json
import time

from _development_budget import check_args
from _development_heads_multi import configure as configure_heads
from _development_lengths import mode_for as lengths_for
from _development_locks import busy, lock_paths
from _development_process import launch
from _development_search import configure
from _development_statistics import order, paired
from _development_system import LABEL, observations
from _measurement import append
from _quality.prefilter_inputs import corpus_names


def job_for(cell, prepared, count, tag, owner, batch=None):
    model = prepared["models"][cell["model"]]
    return {
        **cell,
        "action": "measure",
        "artifact": model["artifacts"][cell["table_dtype"]],
        "corpus": prepared["corpora"][cell["corpus"]],
        "query_corpora": {
            n: prepared["corpora"][n] for n in corpus_names(cell) if n != cell["corpus"]
        },
        "texts": count,
        "tag": tag,
        "bench_owner": owner,
        "batch": batch,
    }


def wait_for_arm(args, ledger, cell_id):
    _, heavy = lock_paths()
    while True:
        check_args(args)
        if ledger.stopping:
            raise InterruptedError("stopped before next development arm")
        _session_gap(args, cell_id, ledger)
        observed = observations(args.work)
        append(args.results / "preflights.jsonl", {**observed, "heavy_busy": busy(heavy)})
        if not busy(heavy) and observed["load"][0] < 5:
            _session_gap(args, cell_id, ledger)
            return
        ledger.stage = "waiting_for_load_or_heavy_work"
        ledger.progress()
        time.sleep(1)


def arm(
    args, cell, prepared, ledger, count, owner, phase, pair, position, name, control, batch=None
):
    wait_for_arm(args, ledger, cell["id"])
    ledger.current, ledger.stage = cell["id"], phase
    ledger.progress()
    tag = f"{cell['id']}-{phase}-{pair}-{position}"
    job = job_for(cell, prepared, count, tag, owner, batch)
    from _development_ids import mode_for

    job["doc_ids"] = mode_for(cell, candidate=not control and name == "candidate")
    job["lengths"] = lengths_for(cell, candidate=not control and name == "candidate")
    configure(job, cell, name, control)
    configure_heads(job, cell, name, control)
    job["cpu_affinity"] = args.cpu_affinity
    job["functional_only"] = args.diagnostic_rehearsal
    job["descriptive_only"] = phase == "descriptive"
    job["sizing_only"] = phase.startswith("size-")
    if job["sizing_only"]:
        job["warmup_seconds"] = 0
    if args.diagnostic_rehearsal:
        job.update(warmup_calls=1, warmup_seconds=0)
    python = args.base_python if control or name == "base" else args.candidate_python
    job["session_last_operation_at_unix"] = _session_gap(args, cell["id"], ledger)
    record = launch(args, job, python, tag, timed=True)
    record.update(
        phase=phase,
        pair=pair,
        position=position,
        arm=name,
        texts=count,
        completed_at_unix=time.time(),
        sizing_only=job["sizing_only"],
        valid_for_comparison=not (
            job["sizing_only"] or job["descriptive_only"] or args.diagnostic_rehearsal
        ),
    )
    append(args.results / "operations.jsonl", {"cell_id": cell["id"], **record})
    return record


def _session_gap(args, cell_id, ledger):
    path = args.results / "operations.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed = [row["completed_at_unix"] for row in records if row["cell_id"] == cell_id]
    inherited = ledger.reused_completed_at.get(cell_id)
    last = max(completed, default=inherited)
    if last is not None and time.time() - last >= 900:
        raise TimeoutError(
            "measurement session interrupted for fifteen minutes; fresh A/A required"
        )
    return last


def block(
    args,
    cell,
    prepared,
    ledger,
    count,
    owner,
    phase,
    identities,
    control=False,
    batch=None,
    budget=None,
):
    rows, proofs = [], set()
    ledger.stage, ledger.current = phase, cell["id"]
    for number in range(cell["pairs"]):
        for position, name in enumerate(order(cell, number)):
            _budget(budget)
            row = arm(
                args,
                cell,
                prepared,
                ledger,
                count,
                owner,
                phase,
                number,
                position,
                name,
                control,
                batch,
            )
            _validate(row, name, identities, control, cell, args.diagnostic_rehearsal)
            proofs.add(json.dumps(row["output"], sort_keys=True))
            rows.append(row)
    if len(proofs) != 1:
        raise ValueError("complete paired stage output differs between builds or rounds")
    return {**LABEL, **paired(cell, rows), "output": rows[0]["output"]}


def _budget(remaining):
    if remaining is not None and remaining() <= 0:
        raise TimeoutError("active calibration budget exhausted before next arm")


def _validate(row, name, identities, control, cell, functional):
    expected = identities[0] if control or name == "base" else identities[1]
    if row["identity"] != expected:
        raise ValueError("build identity differs from the full quality gate")
    if not functional and row["seconds"] < max(cell["minimum_seconds"], cell["target_seconds"]):
        raise ValueError("measured call is shorter than the frozen duration floor")
