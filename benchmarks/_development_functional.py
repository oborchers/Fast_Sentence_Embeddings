"""The functional stage of a development study: a really reduced input and its resource plans.

``--functional`` proves that every path of a study works. It is no speed evidence. It builds a
smaller input (the rows of a corpus chosen by ``_harness.functional``, with the quality gate and
every later step running on exactly those rows) and writes two plans that never mix: the
production plan (the full study) in ``production-plan.json`` and the plan of the run that
happens in ``functional-plan.json``. A cell may declare ``functional_rows``; a path with a
domain (the int8 prefilter needs 65,536 selected rows) never gets fewer rows than it needs.
"""

from __future__ import annotations

import math
from pathlib import Path

from _development_budget import stopped
from _development_identity import STAGE
from _development_locks import coordinated
from _development_process import checked, launch
from _development_system import LABEL
from _harness import functional
from _harness.records import append, atomic, document, sha256
from _quality.prefilter_inputs import corpus_names, search_enabled

DEFAULT_ROWS = 12_000
ROUND_TO = 5_000
PREFILTER_MIN_ROWS = 65_536  # fse._store._prefilter.MIN_ROWS; the path evidence shows a mismatch


def floor_rows(cell):
    """The rows of the cell's corpus below which its path is no longer entered."""
    if not search_enabled(cell):
        return 0
    share = cell["filter_fraction"] if cell["filter_kind"] == "rows" else 1.0
    if share <= 0:  # an empty selection never enters the prefilter, whatever the rows
        return 0
    rows = math.ceil(PREFILTER_MIN_ROWS / share)
    while int(rows * share) < PREFILTER_MIN_ROWS:  # the selected rows are rounded down
        rows += 1
    while rows > 1 and int((rows - 1) * share) >= PREFILTER_MIN_ROWS:
        rows -= 1
    return rows


def validate(cell):
    declared = cell.get("functional_rows")
    if declared is None:
        return
    if type(declared) is not int or declared <= 0:
        raise ValueError("functional_rows must be a positive integer")
    if declared < floor_rows(cell):
        raise ValueError(
            f"functional_rows {declared} is below the {floor_rows(cell)} rows the path needs"
        )


def target_rows(cell):
    """The rows the functional stage keeps of the cell's own corpus."""
    declared = cell.get("functional_rows")
    if declared is not None:
        return declared
    floor = floor_rows(cell)
    return max(DEFAULT_ROWS, math.ceil(floor / ROUND_TO) * ROUND_TO)


def targets(cells):
    """``{corpus: (rows to keep, floor)}``: the largest need of the cells that read it."""
    result = {}
    for cell in cells:
        if cell["stage"] == STAGE:
            continue
        for name in corpus_names(cell):
            own = name == cell["corpus"]
            rows, floor = (target_rows(cell), floor_rows(cell)) if own else (DEFAULT_ROWS, 0)
            old = result.get(name, (0, 0))
            result[name] = (max(rows, old[0]), max(floor, old[1]))
    return result


def reduce_child(job):
    """In its own process: write the reduced corpus file and say what was kept."""
    corpus = job["corpus"]
    data = document(checked(corpus["path"], corpus["sha256"]))
    texts = data["texts"]
    reason = None
    if data.get("score_slices"):
        positions, reason = list(range(len(texts))), "rows carry scores or roles and stay whole"
    else:
        positions = functional.select(len(texts), job["rows"])
    record = functional.reduction(len(texts), positions, floor=job["floor"], reason=reason)
    if not record["reduced"]:
        return {**LABEL, "reduction": record}
    timing = data.get("timing_texts")
    reduced = {
        **data,
        "texts": [texts[i] for i in positions],
        "source": {**data["source"], "functional": record},
    }
    if timing is not None:
        reduced["timing_texts"] = [timing[i] for i in positions]
    path = Path(job["work"]) / (job["name"] + "-functional.json")
    atomic(path, reduced)
    kept = reduced.get("timing_texts", reduced["texts"])
    return {
        **LABEL,
        "reduction": record,
        "corpus": {
            "path": str(path),
            "sha256": sha256(path),
            "texts": len(reduced["texts"]),
            "timing_utf8_bytes": sum(len(text.encode("utf-8")) for text in kept),
            "source": reduced["source"],
        },
    }


def reduce_prepared(args, ledger, cells, prepared):
    """The prepared inputs with every reducible corpus replaced by its reduced file."""
    corpora, records = dict(prepared["corpora"]), {}
    for name, (rows, floor) in sorted(targets(cells).items()):
        corpus = prepared["corpora"][name]
        if corpus["texts"] <= rows:
            keep = list(range(corpus["texts"]))
            reason = "already within the target"
            records[name] = functional.reduction(len(keep), keep, floor=floor, reason=reason)
            continue
        job = {"action": "functional-reduce", "name": name, "corpus": corpus}
        job.update(rows=rows, floor=floor)
        receipt = _reduce(args, ledger, job)
        records[name] = receipt["reduction"]
        if receipt["reduction"]["reduced"]:
            corpora[name] = {**corpus, **receipt["corpus"], "functional": receipt["reduction"]}
    return {
        **prepared,
        "corpora": corpora,
        "functional": {"rule": functional.RULE, "corpora": records},
    }


def _reduce(args, ledger, job):
    """The child runs under the heavy role, so it never overlaps a timing window."""

    def notify(row):
        append(args.results / "coordination.jsonl", row)

    with coordinated(
        "heavy", args.worker_name, "reduce the input", notify, lambda: stopped(ledger)
    ):
        return launch(args, job, args.base_python, "functional-" + job["name"])


def stage(args, ledger, cells, prepared, planner, reducible=True):
    """Write the production plan; under ``--functional`` reduce the input, write the plan of the
    run and return the inputs the run uses. Without it the inputs come back unchanged."""
    production = {
        **LABEL,
        "kind": "production",
        "covers": "the full study",
        "functional_run": args.functional,
        "cells": planner(args, cells, prepared, "production"),
    }
    atomic(args.results / "production-plan.json", production)
    if not args.functional:
        return prepared
    used = reduce_prepared(args, ledger, cells, prepared) if reducible else prepared
    atomic(
        args.results / "functional-plan.json",
        {
            **LABEL,
            "kind": "functional",
            "covers": "this functional run only; " + functional.NOTE,
            "production_plan": "production-plan.json",
            "inputs": used.get("functional", {"rule": None, "corpora": {}}),
            "cells": planner(args, cells, used, "functional"),
        },
    )
    atomic(args.results / "functional-inputs.json", used)
    return used
