"""The parent of the DR-255 probe: admission, one child per cell, outputs and the decision."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from _development_budget import check_args, stopped
from _development_data import prepare
from _development_functional import stage
from _development_locks import coordinated
from _development_process import launch
from _development_records import record_result, session
from _development_system import LABEL
from _harness import resources
from _harness.budget import WallBudgetExpiredError
from _harness.records import append, atomic

from _quality.binary2_decision import decide as decide_2
from _quality.binary_decision import decide
from _quality.binary_inputs import CORPUS, CORPUS_ROWS, PROFILE_2, PROFILES


def admission(cell, prepared, work, requested_limit, kind):
    """Peak bytes of the child (store, its float32 copy, Hamming words) against 85 percent of RAM."""
    limit = resources.require_limit(requested_limit)
    dim = prepared["models"][cell["model"]]["dimensions"]
    corpus = prepared["corpora"][CORPUS]
    table = Path(prepared["models"][cell["model"]]["artifacts"]["float32"]["path"]).stat().st_size
    components = {
        "reserve": 2 * 1024**3,
        "model_copies": 3 * table,
        "store_copy_bits_and_xor": CORPUS_ROWS * (4 * dim) * 3 + CORPUS_ROWS * dim // 8 * 2,
        "python_texts": corpus["timing_utf8_bytes"] * 8,
    }
    disk = CORPUS_ROWS * 4 * dim + corpus["timing_utf8_bytes"] * 4 + 3 * table
    if cell["profile"] == PROFILE_2:  # an int8 store, its codes, the unpacked bits and the sums
        components["int8_store_and_variant_sums"] = (
            CORPUS_ROWS * (dim + 4) * 2 + 8192 * dim * 8 + CORPUS_ROWS * 64 * 4
        )
        disk += CORPUS_ROWS * (dim + 4)
    return resources.assess(
        components, limit=limit, work=work, disk=disk, disk_fraction=0.85, kind=kind, **LABEL
    )


def plans(args, cells, prepared, kind):
    return {
        cell["id"]: admission(cell, prepared, args.work, args.memory_limit_bytes, kind)
        for cell in cells
    }


def _job(cell, prepared):
    return {
        "action": "binary-recall-2" if cell["profile"] == PROFILE_2 else "binary-recall",
        "queries": cell["queries_per_set"],
        "model": cell["model"],
        "workers": cell["workers"],
        "artifact": prepared["models"][cell["model"]]["artifacts"][cell["table_dtype"]],
        "corpus": prepared["corpora"][CORPUS],
    }


def count_cell(args, cell, prepared, ledger):
    kind = "functional" if args.functional else "production"
    plan = admission(cell, prepared, args.work, args.memory_limit_bytes, kind)
    append(args.results / "memory.jsonl", {"cell_id": cell["id"], "phase": "binary", **plan})
    if not plan["admitted"]:
        raise MemoryError("the binary recall probe exceeds RAM or disk admission")

    def notify(row):
        append(args.results / "coordination.jsonl", row)

    with coordinated(
        "heavy", args.worker_name, "binary recall probe", notify, lambda: stopped(ledger)
    ):
        receipt = launch(args, _job(cell, prepared), args.candidate_python, cell["id"] + "-recall")
    append(
        args.results / "checks.jsonl",
        {"cell_id": cell["id"], **receipt["checks"], "input_identity": receipt["input_identity"]},
    )
    for record in receipt["records"]:
        append(args.results / "binary.jsonl", {"cell_id": cell["id"], **record})
    return receipt


def _preflight(args):
    """A conservative check before the preparation downloads and exports anything."""
    limit, _ = resources.require_limit(args.memory_limit_bytes)
    free = shutil.disk_usage(args.work.parent).free
    if limit * 0.85 < 4 * 1024**3 or free * 0.85 < 4 * 1024**3:
        raise MemoryError("less than 4 GiB RAM or 4 GiB disk for the preparation and the probe")


def selected(cells):
    """Whether the cells are of a profile of this probe; one profile runs alone."""
    profiles = {cell["profile"] for cell in cells}
    if profiles & set(PROFILES) and (len(profiles) > 1 or not profiles <= set(PROFILES)):
        raise ValueError(f"{PROFILES} cells run alone, one profile at a time")
    return bool(profiles & set(PROFILES))


def run_cells(args, spec):
    cells = spec["cells"]
    with session(args, spec) as ledger:
        _preflight(args)
        prepared = stage(args, ledger, cells, prepare(args, cells, ledger), plans, reducible=False)
        check_args(args)
        for cell in cells:
            if ledger.stopping:
                break
            ledger.current, ledger.stage = cell["id"], "binary_recall"
            started = time.monotonic()
            try:
                receipt = count_cell(args, cell, prepared, ledger)
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
            except MemoryError as error:  # refused before the child began
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
                ledger, cell, started, status="valid", binary={"records": len(receipt["records"])}
            )
    failed = any(r["status"] == "failed" for r in ledger.finished)
    judge = decide_2 if cells[0]["profile"] == PROFILE_2 else decide
    outcome = judge(args.results / "binary.jsonl", cells, ledger.finished, failed)
    atomic(args.results / "decision.json", {**LABEL, **outcome})
    print(json.dumps(outcome), flush=True)
    return ledger.done()
