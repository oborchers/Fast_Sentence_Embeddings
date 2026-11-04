"""Fail-fast full byte gates for installed builds through quality.py."""

from __future__ import annotations

import struct
import time

from _development_budget import WallBudgetExpiredError, check_args, stopped
from _development_config import load
from _development_data import prepare
from _development_ids import mode_for
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
        "heavy", args.worker_name, "full output quality gate", notify, lambda: stopped(ledger)
    ):
        builds = (
            [("candidate", args.candidate_python)]
            if getattr(args, "local_phase", None) == "descriptive"
            else [("base", args.base_python), ("candidate", args.candidate_python)]
        )
        for name, python in builds:
            job = dict(
                action="quality",
                artifact=model["artifacts"][cell["table_dtype"]],
                corpus=corpus,
                workers=cell["workers"],
                output=str(args.work / (tag + "-" + name)),
                doc_ids=mode_for(cell, candidate=name == "candidate"),
                id_chunks=cell["id_chunks"],
            )
            records.append(launch(args, job, python, tag + "-" + name))
            if records[-1]["document_ids"] != {"mode": job["doc_ids"], "alignment_checked": True}:
                raise ValueError(
                    "quality document-ID alignment proof differs from the requested mode"
                )
        bits = compare(records[0], records[-1])
    if cell["corpus"] == "stsb" and set(bits) != {"train", "dev", "test"}:
        raise ValueError("complete STS-B score splits required")
    result = {
        **LABEL,
        "bit_identical": True,
        "scope": "within_build_direct_store"
        if len(records) == 1
        else "base_candidate_direct_store",
        "score_bits": bits,
        "identities": [r["identity"] for r in records],
        "document_ids": [r["document_ids"] for r in records],
    }
    append(args.results / "checks.jsonl", {"cell_id": cell["id"], **result})
    return result


def run(args):
    spec = load(args.development_input, args.cells)
    spec["functional_only"] = args.diagnostic_rehearsal
    seen, cells = set(), []
    for cell in spec["cells"]:
        key = tuple(
            cell[name]
            for name in (
                "model",
                "corpus",
                "table_dtype",
                "workers",
                "profile",
                "base_doc_ids",
                "candidate_doc_ids",
                "id_chunks",
            )
        )
        if key not in seen:
            seen.add(key)
            cells.append(cell)
    spec["cells"] = cells
    with session(args, spec) as ledger:
        prepared = prepare(args, cells, ledger)
        check_args(args)
        for cell in cells:
            if ledger.stopping:
                break
            ledger.current, ledger.stage = cell["id"], "full_output_quality_gate"
            started = time.monotonic()
            try:
                result = quality_gate(args, cell, prepared, ledger, cell["id"] + "-quality")
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
            except Exception as error:
                record_result(ledger, cell, started, status="failed", reason=repr(error))
                break
            record_result(ledger, cell, started, status="valid", quality=result)
    return ledger.done()
