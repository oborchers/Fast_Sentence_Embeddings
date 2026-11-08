"""The parent of the prefilter count: admission, one child per cell, outputs and the decision."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from _development_budget import check_args, stopped
from _development_data import prepare
from _development_locks import coordinated
from _development_process import launch
from _development_records import record_result, session
from _development_system import LABEL, memory_limit
from _harness.budget import WallBudgetExpiredError
from _harness.records import append, atomic

from _quality.prefilter_decision import decide
from _quality.prefilter_inputs import PROFILE, corpus_names

BLOCK_ROWS, QUERY_BLOCK = 16_384, 32


def admission(cell, prepared, work, requested_limit):
    """Peak bytes of every phase against 85 percent of RAM, and the disk of the stores."""
    limit = memory_limit()
    if requested_limit is not None:
        limit = min(limit, requested_limit) if limit else requested_limit
    if not limit:
        raise ValueError("RAM cannot be detected; supply --memory-limit-bytes")
    corpus = prepared["corpora"][cell["corpus"]]
    model = prepared["models"][cell["model"]]
    dim, rows = model["dimensions"], max(cell["store_sizes"])
    texts = corpus["timing_utf8_bytes"] + corpus["texts"] * 64
    table = Path(model["artifacts"]["float32"]["path"]).stat().st_size
    base = 1024**3 + texts + 2 * table
    phases = {
        "store_build": base + 4 * 64 * 1024**2 + rows * 8,
        "count": base
        + rows * (dim + 4)  # the mapped store, file backed
        + rows * QUERY_BLOCK * (8 + 4)  # a (float64) and c (float32)
        + 3 * QUERY_BLOCK * rows  # the kept masks of every k
        + 2 * BLOCK_ROWS * dim * 8,  # a float64 row block and its products
    }
    disk = rows * (dim + 4) * 2 + texts + 3 * table + (1024**3 if cell["scope"] == "main" else 0)
    free = shutil.disk_usage(work).free
    peak = max(phases.values())
    return {
        **LABEL,
        "phases": phases,
        "estimated_peak_bytes": peak,
        "memory_limit_bytes": limit,
        "safety_fraction": 0.85,
        "required_disk_bytes": disk,
        "available_disk_bytes": free,
        "admitted": peak <= limit * 0.85 and free * 0.85 >= disk,
    }


def _job(cell, prepared):
    model = prepared["models"][cell["model"]]
    return {
        "action": "prefilter-count",
        **{
            name: cell[name]
            for name in (
                "model",
                "scope",
                "store_sizes",
                "queries_per_set",
                "k_values",
                "probe_rows",
                "probe_queries",
                "workers",
            )
        },
        "artifact": model["artifacts"][cell["table_dtype"]],
        "corpus": prepared["corpora"][cell["corpus"]],
        "query_corpora": {
            n: prepared["corpora"][n] for n in corpus_names(cell) if n != cell["corpus"]
        },
    }


def _write(args, cell, receipt):
    append(
        args.results / "checks.jsonl",
        {"cell_id": cell["id"], "input_identity": receipt["input_identity"]},
    )
    parts = receipt["synthetic"] + receipt["stores"]
    for part in parts:
        append(args.results / "checks.jsonl", {"cell_id": cell["id"], **part["checks"]})
        for record in part["records"]:
            append(args.results / "prefilter.jsonl", {"cell_id": cell["id"], **record})
        for row in part.get("queries", []):
            append(args.results / "queries.jsonl", {"cell_id": cell["id"], **row})


def count_cell(args, cell, prepared, ledger):
    plan = admission(cell, prepared, args.work, args.memory_limit_bytes)
    append(args.results / "memory.jsonl", {"cell_id": cell["id"], "phase": "prefilter", **plan})
    if not plan["admitted"]:
        raise MemoryError("the prefilter count exceeds RAM or disk admission")

    def notify(row):
        append(args.results / "coordination.jsonl", row)

    with coordinated(
        "heavy", args.worker_name, "int8 prefilter count", notify, lambda: stopped(ledger)
    ):
        receipt = launch(args, _job(cell, prepared), args.candidate_python, cell["id"] + "-count")
    _write(args, cell, receipt)
    return receipt


def _preflight(args, cells):
    """A conservative check before the preparation downloads and writes anything."""
    limit = memory_limit()
    if args.memory_limit_bytes is not None:
        limit = min(limit, args.memory_limit_bytes) if limit else args.memory_limit_bytes
    if not limit:
        raise ValueError("RAM cannot be detected; supply --memory-limit-bytes")
    free = shutil.disk_usage(args.work.parent).free
    if any(c["scope"] == "main" for c in cells) and (
        limit * 0.85 < 4 * 1024**3 or free * 0.85 < 6 * 1024**3
    ):
        raise MemoryError("less than 4 GiB RAM or 6 GiB disk for the preparation and the counts")


def selected(cells):
    """Whether the cells are of the profile; it runs alone."""
    profiles = {cell["profile"] for cell in cells}
    if PROFILE in profiles and len(profiles) > 1:
        raise ValueError(f"{PROFILE} cells run alone")
    return PROFILE in profiles


def run_cells(args, spec):
    cells = spec["cells"]
    with session(args, spec) as ledger:
        _preflight(args, cells)
        prepared = prepare(args, cells, ledger)
        check_args(args)
        done = {}
        for cell in cells:
            if ledger.stopping:
                break
            ledger.current, ledger.stage = cell["id"], "prefilter_count"
            started = time.monotonic()
            try:
                done[cell["id"]] = count_cell(args, cell, prepared, ledger)
                check_args(args)
            except WallBudgetExpiredError as error:
                record_result(
                    ledger,
                    cell,
                    started,
                    status="inconclusive",
                    decision="inconclusive_budget",
                    reason=str(error),
                )
                ledger.stopping = True
                break
            except MemoryError as error:  # refused before any counting began
                record_result(
                    ledger,
                    cell,
                    started,
                    status="not_started",
                    decision="not_run",
                    reason=str(error),
                )
                ledger.stopping = True
                break
            except Exception as error:
                record_result(ledger, cell, started, status="failed", reason=repr(error))
                break
            record_result(
                ledger,
                cell,
                started,
                status="valid",
                scope=cell["scope"],
                prefilter={"stores": len(done[cell["id"]]["stores"])},
            )
    _decision(args, cells, ledger)
    return ledger.done()


def _decision(args, cells, ledger):
    failed = any(r["status"] == "failed" for r in ledger.finished)
    outcome = decide(args.results / "prefilter.jsonl", cells, ledger.finished, failed)
    atomic(args.results / "decision.json", {**LABEL, **outcome})
    print(json.dumps(outcome), flush=True)
