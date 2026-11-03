"""Fail-fast full byte gates for installed builds through quality.py."""

from __future__ import annotations

import struct
import time

from _development_config import load
from _development_data import prepare
from _development_locks import coordinated
from _development_process import checked, launch
from _development_records import record_result, session
from _development_resources import admission
from _development_system import LABEL
from _measurement import append


def same_bytes(first, second):
    import numpy as np

    a = np.load(checked(first["path"], first["sha256"]), mmap_mode="r", allow_pickle=False)
    b = np.load(checked(second["path"], second["sha256"]), mmap_mode="r", allow_pickle=False)
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError("quality output shape or dtype differs")
    if list(a.shape) != first["shape"] or a.dtype.str != first["dtype"]:
        raise ValueError("quality output differs from its receipt")
    if list(b.shape) != second["shape"] or b.dtype.str != second["dtype"]:
        raise ValueError("quality output differs from its receipt")
    for begin in range(0, len(a), 4096):
        av, bv = a[begin : begin + 4096], b[begin : begin + 4096]
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError("nonfinite quality output")
        different = np.flatnonzero(av.view(np.uint8).reshape(-1) != bv.view(np.uint8).reshape(-1))
        if different.size:
            offset = begin * a[0:1].nbytes + int(different[0])
            raise ValueError(f"quality bytes differ at byte {offset}: {second['path']}")


def _tensors(first, second):
    if set(first) != set(second):
        raise ValueError("quality tensor inventory differs")
    for name in sorted(first):
        same_bytes(first[name], second[name])


def compare(base, candidate):
    if base["encoder_fingerprint"] != candidate["encoder_fingerprint"]:
        raise ValueError("quality encoder fingerprints differ")
    if set(base["outputs"]) != {"float32", "float16", "int8"} or set(candidate["outputs"]) != set(
        base["outputs"]
    ):
        raise ValueError("quality formats differ")
    for dtype, outputs in base["outputs"].items():
        other = candidate["outputs"][dtype]
        if set(outputs) != {"direct", "stored"} or set(other) != set(outputs):
            raise ValueError("quality output paths differ")
        _tensors(outputs["direct"], outputs["stored"])
        _tensors(other["direct"], other["stored"])
        for mode in outputs:
            _tensors(outputs[mode], other[mode])
    return _score_bits(base["scores"], candidate["scores"])


def _score_bits(first, second):
    if set(first) != set(second):
        raise ValueError("quality score splits differ")
    result = {}
    for split, scores in first.items():
        if scores["pairs"] != second[split]["pairs"]:
            raise ValueError("STS-B pair counts differ")
        result[split] = {}
        for metric in ("pearson", "spearman"):
            bits = struct.pack("<d", scores[metric])
            if bits != struct.pack("<d", second[split][metric]):
                raise ValueError(f"STS-B {split} {metric} float64 bits differ")
            result[split][metric] = {"value": scores[metric], "float64_le_hex": bits.hex()}
    return result


def quality_gate(args, cell, prepared, ledger, tag):
    model = prepared["models"][cell["model"]]
    corpus = prepared["corpora"][cell["corpus"]]
    plan = admission(
        {**cell, "stage": "encode-memory"},
        prepared,
        corpus["texts"],
        args.work,
        args.memory_limit_bytes,
    )
    append(args.results / "memory.jsonl", {"cell_id": cell["id"], "phase": "quality", **plan})
    if not plan["admitted"]:
        raise MemoryError("complete quality corpus exceeds RAM or disk admission")
    records = []

    def notify(row):
        append(args.results / "coordination.jsonl", row)

    with coordinated(
        "heavy", args.worker_name, "full output quality gate", notify, lambda: ledger.stopping
    ):
        for name, python in (("base", args.base_python), ("candidate", args.candidate_python)):
            job = dict(
                action="quality",
                artifact=model["artifacts"][cell["table_dtype"]],
                corpus=corpus,
                workers=cell["workers"],
                output=str(args.work / (tag + "-" + name)),
            )
            records.append(launch(args, job, python, tag + "-" + name))
        bits = compare(*records)
    if cell["corpus"] == "stsb" and set(bits) != {"train", "dev", "test"}:
        raise ValueError("complete STS-B score splits required")
    result = {
        **LABEL,
        "bit_identical": True,
        "score_bits": bits,
        "identities": [r["identity"] for r in records],
    }
    append(args.results / "checks.jsonl", {"cell_id": cell["id"], **result})
    return result


def run(args):
    spec = load(args.development_input, args.cells)
    spec["functional_only"] = args.diagnostic_rehearsal
    seen, cells = set(), []
    for cell in spec["cells"]:
        key = cell["model"], cell["corpus"], cell["table_dtype"], cell["workers"]
        if key not in seen:
            seen.add(key)
            cells.append(cell)
    spec["cells"] = cells
    with session(args, spec) as ledger:
        prepared = prepare(args, cells, ledger)
        for cell in cells:
            if ledger.stopping:
                break
            ledger.current, ledger.stage = cell["id"], "full_output_quality_gate"
            started = time.monotonic()
            result = quality_gate(args, cell, prepared, ledger, cell["id"] + "-quality")
            record_result(ledger, cell, started, status="valid", quality=result)
    return ledger.done()
