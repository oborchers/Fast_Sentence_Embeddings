"""Absolute Sentence Transformers observations, with no fse arm or paired factor."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from _harness.records import append
from _speed_cells import check_output
from _speed_versions import pair_rounds


def validate(cell):
    """Keep device and precision explicit; the first variant supports CPU only."""
    if cell.get("device") != "cpu":
        raise ValueError("transformer device must be 'cpu'")
    if cell.get("backend") != "st" or cell.get("reference") is not None:
        raise ValueError("transformer requires backend 'st' without a reference")
    if cell.get("torch_dtype") != "float32" or cell.get("dtype") != "float32":
        raise ValueError("transformer requires float32 weights and output")
    for field in ("texts", "workers", "dimensions", "batch_size", "input_prefix_texts"):
        if type(cell.get(field)) is not int or cell[field] <= 0:
            raise ValueError(f"transformer {field} must be a positive integer")
    if cell["input_prefix_texts"] < cell["texts"] or cell.get("input") != "st":
        raise ValueError("transformer requires a prefix of the st input")
    pair_rounds(cell)


def prepare_prefix(path, source, mandatory, cell):
    from _speed_cells import write_input

    parent = cell["texts"] if cell.get("functional_only") else cell["input_prefix_texts"]
    return write_input(path, source, mandatory, parent, 0, prefix_texts=cell["texts"])


def sample_check(job, texts, warm):
    """Compare the pool with the native single-process path before the timer."""
    from _speed_pipeline import st_encoder

    reference = st_encoder(dict(job, workers=1))
    # Include the eight mandatory UNK paragraphs, plus eight sampled sentences.
    check_output(reference(texts[:16]), warm[:16], exact=False)


def summary(cell, rows, identity):
    if len(rows) != pair_rounds(cell):
        raise ValueError("incomplete transformer repetitions")
    if len({json.dumps(r["identity"], sort_keys=True) for r in rows}) != 1:
        raise ValueError("transformer package identity changed between repetitions")
    seconds = [r["seconds"] for r in rows]
    if any(not math.isfinite(t) or t <= 0 for t in seconds):
        raise ValueError("transformer duration must be positive and finite")
    short = not cell.get("functional_only") and any(t < 2 for t in seconds)
    rates = [cell["texts"] / t for t in seconds]
    q1, _, q3 = statistics.quantiles(rates, n=4, method="inclusive")
    return {
        "protocol": "transformer@1",
        "status": "invalid" if short else "valid",
        "reason": "call shorter than two seconds" if short else None,
        "input_identity": identity,
        "sample_checked": True,
        "device": cell["device"],
        "torch_dtype": cell["torch_dtype"],
        "batch_size": cell["batch_size"],
        "rounds_completed": len(rows),
        "texts_per_second": None if short else {"median": statistics.median(rates), "iqr": q3 - q1},
    }


def run(context, cell):
    from _speed_run import cleanup, launch, prepare_input, sample_array

    validate(cell)
    path, sample, identity = prepare_input(context, cell)
    rows = []
    for number in range(pair_rounds(cell)):
        if not context["ledger"].admit(cell["estimated_seconds"] / cell["rounds"]):
            return {
                "protocol": "transformer@1",
                "status": "partial",
                "reason": "budget or stop signal",
            }
        row = launch(context, cell, "st", number, path, sample, identity)
        rows.append(row)
        # The child compares native single-process and pool output before timing.
        if not row.get("transformer_sample_checked"):
            raise ValueError("transformer child has no sample check")
        append(
            context["ledger"].root / "checks.jsonl",
            {
                "cell_id": cell["id"],
                "round": number,
                "native_sample_checked": True,
                "rows": len(sample_array(row)),
            },
        )
        cleanup(context, [Path(row["sample_file"])])
    result = summary(cell, rows, identity)
    cleanup(context, [path, path.with_suffix(".json"), sample])
    return result
