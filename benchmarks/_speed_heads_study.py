"""Admission, sequential arms and summaries for publication head kinds and sinks."""

from __future__ import annotations

from pathlib import Path

from _harness import resources
from _harness.records import append, sha256
from _speed_cells import check_output
from _speed_protocol import protocol
from _speed_versions import pair_rounds


def validate(cell):
    for name in (
        "texts",
        "workers",
        "dimensions",
        "heads",
        "vector_dimensions",
        "planned_input_bytes",
        "estimated_seconds",
        "rounds",
    ):
        if type(cell.get(name)) is not int or cell[name] <= 0:
            raise ValueError(f"head {name} must be a positive integer")
    if cell["dtype"] != "float32" or cell["vector_dimensions"] != 256:
        raise ValueError("publication heads require float32 POTION vectors of width 256")
    if cell["model"] != "minishlab/potion-base-8M":
        raise ValueError("publication heads require the pinned POTION-base-8M")
    validate_kind(cell)
    validate_workload(cell)


def validate_kind(cell):
    if cell["kind"] == "heads-kind":
        if cell.get("head_kind") not in ("multilabel", "calibrated"):
            raise ValueError("head_kind must be multilabel or calibrated")
        if cell["heads"] != 1 or cell["dimensions"] != 4 or cell["reference"] is not None:
            raise ValueError("head kinds require one four-output head and no reference arm")
    elif cell["heads"] != 4 or cell["dimensions"] != 16 or cell["reference"] != "fse-two-pass":
        raise ValueError("head sinks require four heads against fse-two-pass")


def validate_workload(cell):
    pair_rounds(cell)
    expected = (5_000_000, 1, 5) if cell["kind"] == "heads-kind" else (10_000_000, 16, 3)
    prefix = "production_" if cell.get("functional_only") else ""
    actual = (cell.get(prefix + "texts"), cell.get(prefix + "workers"), cell["rounds"])
    if actual != expected:
        raise ValueError(
            "head publication texts, workers and rounds differ from the fixed protocol"
        )


def memory_plan(cell, model_bytes, limit, retained=0, *, work=None):
    """Include input pages, folded plans, source retention and one set of stores."""
    validate(cell)
    n, d, w = cell["texts"], cell["vector_dimensions"], cell["workers"]
    sinks = cell["kind"] == "heads-sinks"
    functional = cell.get("functional_only", False)
    # The shared POTION input caps every text at 3072 characters (at most four UTF-8 bytes each).
    input_bytes = n * (3072 * 4 + 1) if functional else cell["planned_input_bytes"]
    components = {
        "controller_libraries_reserve": (2 if functional else 8) * resources.GIB,
        "retained_source": retained,
        "input_pages_and_preparation": input_bytes + n * 8 * 4,
        "document_ids": n * 8 if sinks else 0,
        "models_and_plans": model_bytes * (w + 1) * 2,
        "worker_buffers": w * 256 * 1024**2,
        "sample_buffers": min(n, 10_000) * (d + cell["dimensions"]) * 4 * 4,
        "one_store_set_pages" if sinks else "output_texts_and_scratch": n
        * (4 * (d + cell["dimensions"]) + 20)
        if sinks
        else n * cell["dimensions"] * 8 + input_bytes * 5 + n * 72,
    }
    disk = input_bytes + n * (4 * (d + cell["dimensions"]) + 20) + 5 * resources.GIB
    return resources.assess(
        components,
        limit=limit,
        work=work,
        disk=disk if work else None,
        kind="functional" if functional else "production",
    )


def preflight(context, cell, limit):
    from _speed_pipeline import model_bytes

    plan = memory_plan(
        cell,
        model_bytes(context["models"][cell["model"]]),
        limit,
        context["production_source_bytes"],
    )
    return dict(cell_id=cell["id"], **plan)


def compare_samples(rows):
    import numpy as np

    samples = []
    for row in rows:
        path = Path(row["head_sample_file"])
        if sha256(path) != row["head_sample_sha256"]:
            raise ValueError("head sink sample file changed")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"indices", "vectors", "scores"}:
                raise ValueError("head sink sample fields differ")
            samples.append({key: saved[key] for key in saved.files})
    for key in samples[0]:
        check_output(samples[0][key], samples[1][key], exact=True)


def summary(cell, rows, identity):
    from _speed_run import arm_summary, validate_pair_arms

    validate(cell)
    arms = grouped_arms(cell, rows)
    complete = all(len(values) == pair_rounds(cell) for values in arms.values())
    if complete:
        validate_pair_arms(arms, pair_rounds(cell))
    status, reason = summary_status(cell, rows, complete)
    result = {
        "protocol": protocol(cell)["name"],
        "status": status,
        "reason": reason,
        "input_identity": identity,
        "sample_checked": bool(rows),
        "aa": {"status": "not_run"},
        "arms": {name: arm_summary(values) for name, values in arms.items() if values},
    }
    if complete and len(arms) == 2:
        result["paired_speedup"] = paired_factor(arms)
    return result


def grouped_arms(cell, rows):
    labels = ["fse", "fse-two-pass"] if cell["kind"] == "heads-sinks" else ["fse"]
    arms = {label: [] for label in labels}
    for row in rows:
        if row.get("diagnostic_profile"):
            raise ValueError("diagnostic profiles cannot enter head summaries")
        if row["label"] not in arms:
            raise ValueError("unexpected head study arm")
        arms[row["label"]].append(row)
    return arms


def summary_status(cell, rows, complete):
    if not complete:
        return "partial", "budget or stop signal"
    short = not cell.get("functional_only") and any(
        min(r.get("inference_call_seconds", [r["seconds"]])) < 2 for r in rows
    )
    if short:
        return "invalid", "call shorter than two seconds"
    return "valid", None


def paired_factor(arms):
    from _measurement import spread

    return spread(
        [
            b["seconds"] / a["seconds"]
            for a, b in zip(arms["fse"], arms["fse-two-pass"], strict=True)
        ]
    )


def run(context, cell):
    from _speed_pipeline import model_bytes
    from _speed_run import cleanup, prepare_input

    plan = memory_plan(
        cell,
        model_bytes(context["models"][cell["model"]]),
        resources.require_limit(),
        context["retained_source_bytes"],
        work=context["work"],
    )
    append(context["ledger"].root / "memory.jsonl", dict(cell_id=cell["id"], **plan))
    if not plan["admitted"]:
        raise MemoryError("head study exceeds the memory or disk allowance")
    path, sample, identity = prepare_input(context, cell)
    labels = ["fse", "fse-two-pass"] if cell["kind"] == "heads-sinks" else ["fse"]
    rows = []
    try:
        for number in range(pair_rounds(cell)):
            if not context["ledger"].admit(cell["estimated_seconds"] / cell["rounds"]):
                break
            rows.extend(round_arms(context, cell, number, labels, path, sample, identity))
        return summary(cell, rows, identity)
    finally:
        cleanup(context, [path, path.with_suffix(".json"), sample])


def round_arms(context, cell, number, labels, path, sample, identity):
    from _speed_run import cleanup, launch

    current = []
    for label in labels if number % 2 == 0 else labels[::-1]:
        row = launch(context, cell, label, number, path, sample, identity)
        if not row.get("sample_numpy_bitwise_equal") or row["sample_checked_rows"] != min(
            10_000, cell["texts"]
        ):
            raise ValueError("head arm lacks its numpy bit comparison")
        current.append(row)
        if cell["kind"] == "heads-kind":
            cleanup(context, [Path(row["sample_file"])])
    if len(labels) == 2:
        compare_samples(current)
    append(
        context["ledger"].root / "checks.jsonl",
        dict(
            cell_id=cell["id"],
            round=number,
            rows=len(identity["sample_rows"]),
            sample_numpy_bitwise_equal=True,
            cross_arm_bitwise_equal=len(labels) == 2,
        ),
    )
    return current
