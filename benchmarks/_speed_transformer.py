"""Absolute Sentence Transformers observations, with no fse arm or paired factor."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from _harness.pressure import noise, report
from _harness.records import append
from _speed_cells import check_output
from _speed_versions import pair_rounds


def validate(cell):
    """Keep device, precision and fixed input explicit for CPU and CUDA."""
    from _measurement_config import validate_digest

    if cell.get("device") not in ("cpu", "cuda"):
        raise ValueError("transformer device must be cpu or cuda")
    if cell["device"] == "cuda":
        from _speed_gpu import validate as validate_gpu

        validate_gpu(cell)
    if cell.get("backend") != "st" or cell.get("reference") is not None:
        raise ValueError("transformer requires backend 'st' without a reference")
    if cell.get("dtype") != "float32" or (
        cell["device"] == "cpu" and cell.get("torch_dtype") != "float32"
    ):
        raise ValueError("transformer requires float32 weights and output")
    for field in (
        "texts",
        "workers",
        "dimensions",
        "batch_size",
        "input_prefix_texts",
        "planned_input_bytes",
    ):
        if type(cell.get(field)) is not int or cell[field] <= 0:
            raise ValueError(f"transformer {field} must be a positive integer")
    if cell["input_prefix_texts"] < cell["texts"] or cell.get("input") != "st":
        raise ValueError("transformer requires a prefix of the st input")
    validate_digest(cell.get("planned_input_sha256"))
    validate_estimate(cell.get("estimated_seconds"))
    pair_rounds(cell)


def validate_estimate(value):
    if type(value) not in (int, float):
        raise ValueError("transformer estimated_seconds must be a positive finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value <= 0:
        raise ValueError("transformer estimated_seconds must be a positive finite number")


def validate_job(job):
    if job.get("kind") == "transformer":
        validate(job)


def check_warm(job, texts, warm, indices):
    if job.get("kind") != "transformer":
        return None
    sample_check(job, texts, warm)
    return warm[indices].copy()


def check_measured(vectors, sample, warmed):
    if warmed is None:
        return
    import numpy as np

    check_output(warmed, sample, exact=False)
    if not np.isfinite(vectors).all():
        raise ValueError("nonfinite transformer output")


def receipt_fields(job, warmed):
    if warmed is None:
        return {}
    return {
        "transformer_sample_checked": True,
        **{key: job[key] for key in ("device", "torch_dtype", "batch_size")},
    }


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
    if len(rows) != repetitions(cell):
        raise ValueError("incomplete transformer repetitions")
    if len({json.dumps(r["identity"], sort_keys=True) for r in rows}) != 1:
        raise ValueError("transformer package identity changed between repetitions")
    seconds = [r["seconds"] for r in rows]
    if any(not math.isfinite(t) or t <= 0 for t in seconds):
        raise ValueError("transformer duration must be positive and finite")
    short = not cell.get("functional_only") and any(t < 2 for t in seconds)
    if cell.get("gpu_phase") == "probe":
        return {
            "protocol": "transformer@1",
            "status": "valid",
            "gpu_phase": "probe",
            "input_identity": identity,
            "sample_checked": True,
            "speed_evidence": False,
        }
    rates = [cell["texts"] / t for t in seconds]
    return {
        "protocol": "transformer@1",
        "status": "invalid" if short else "valid",
        "reason": "call shorter than two seconds" if short else None,
        "input_identity": identity,
        "cpu_pressure": report(rows),
        "sample_checked": True,
        "device": cell["device"],
        "torch_dtype": cell["torch_dtype"],
        "batch_size": cell["batch_size"],
        "rounds_completed": len(rows),
        "texts_per_second": None if short else {"median": statistics.median(rates), **noise(rates)},
    }


def admit(context, cell, identity):
    from _harness.resources import require_limit
    from _speed_cells import memory_plan, model_bytes

    plan = memory_plan(
        cell,
        identity["file_bytes"],
        model_bytes(context["models"][cell["model"]]),
        require_limit(),
        "functional" if cell.get("functional_only") else "production",
    )
    append(context["ledger"].root / "memory.jsonl", dict(cell_id=cell["id"], **plan))
    if not plan["admitted"]:
        raise MemoryError("transformer input exceeds the actual memory allowance")


def run(context, cell):
    from _speed_run import cleanup, launch, prepare_input, sample_array

    validate(cell)
    path, sample, identity = prepare_input(context, cell)
    admit(context, cell, identity)
    rows = []
    for number in range(repetitions(cell)):
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


def repetitions(cell):
    return 1 if cell.get("gpu_phase") == "probe" else pair_rounds(cell)
