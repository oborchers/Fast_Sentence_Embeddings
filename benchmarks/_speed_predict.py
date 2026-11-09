"""Single-call native-file classification headlines, with owned input and memory plans.

The placeholder is functional evidence only. Publication is refused until a pinned real
head is configured and its loader is implemented. No model is trained by this module.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import _speed_cells as inputs
from _harness import resources
from _harness.records import append, atomic, document, sha256, sync_directory
from _measurement import package_identity, spread, telemetry
from _speed_write import headline_tokens

LABELS = ("toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit")
PROFILE = "toxicity-placeholder@1"


def validate(cell):
    head = cell["head"]
    if cell["output"] not in ("labels", "proba"):
        raise ValueError("predict output must be labels or proba")
    if head["labels"] != list(LABELS):
        raise ValueError("headline requires the six ordered toxicity labels")
    if head["kind"] not in ("placeholder", "artifact"):
        raise ValueError("head kind must be placeholder or artifact")
    if head["kind"] == "placeholder" and (head["profile"] != PROFILE or head["seed"] != 17):
        raise ValueError("placeholder requires toxicity-placeholder@1 and seed 17")
    if head["kind"] == "artifact":
        if not all(head.get(key) for key in ("repo", "revision", "filename", "sha256")):
            raise ValueError("artifact head requires repo, revision, filename and sha256")
        raise ValueError("pinned real-head loading is not implemented yet")
    if cell["rounds"] != 3 or cell["warmup_texts"] <= 0:
        raise ValueError("predict requires three rounds and positive warmup_texts")


def classifier(job):
    import numpy as np

    from fse import Classifier, Encoder

    validate(job)
    if not job.get("functional_only"):
        raise ValueError("placeholder heads are only allowed in functional probes")
    encoder = Encoder.from_pretrained(job["artifact"], normalize=False, mmap=True)
    rng = np.random.default_rng(job["head"]["seed"])
    weight = rng.standard_normal((len(LABELS), encoder.output_dim)).astype(np.float32)
    bias = rng.standard_normal(len(LABELS)).astype(np.float32)
    model = Classifier.from_arrays(
        encoder,
        weight,
        bias,
        labels=LABELS,
        link="multilabel",
        normalized=False,
    )
    if model.path() != "folded" or tuple(model.classes_) != LABELS:
        raise ValueError("headline head must be folded with the ordered labels")
    if dict(model.thresholds) != dict.fromkeys(LABELS, 0.5):
        raise ValueError("placeholder thresholds must be 0.5")
    provenance = dict(
        job["head"],
        link="multilabel",
        normalized=False,
        path=model.path(),
        weight_sha256=hashlib.sha256(weight.tobytes()).hexdigest(),
        bias_sha256=hashlib.sha256(bias.tobytes()).hexdigest(),
        fingerprint=model.fingerprint,
        note="No training; functional placeholder only, not the published toxicity classifier",
    )
    return model, provenance


def memory_plan(cell, model_bytes, input_bytes, warmup_bytes, work, limit, kind):
    n, k, w = cell["texts"], len(cell["head"]["labels"]), cell["workers"]
    components = {
        "controller_libraries_reserve": (2 if cell.get("functional_only") else 8) * resources.GIB,
        "input_sampling_arrays": n * 24,
        "outputs": n * k * (1 if cell["output"] == "labels" else 4),
        "worker_buffers": w * 64 * 1024**2,
        "model_copies": model_bytes * (w + 1),
        "mapped_input": input_bytes,
        "sample_buffers": min(n, 10000) * k * 4 * 3,
    }
    return resources.assess(
        components,
        limit=limit,
        work=work,
        disk=input_bytes + warmup_bytes + 5 * resources.GIB,
        kind=kind,
        output=cell["output"],
        label_count=k,
    )


def admit_plans(context, cell, sizes=None):
    """Record production and actual plans before input, then again with known file sizes."""
    ledger, work = context["ledger"], context["work"]
    model_bytes = Path(context["models"][cell["model"]]["artifact"]).stat().st_size
    production = dict(
        cell,
        texts=cell.get("production_texts", cell["texts"]),
        workers=cell.get("production_workers", cell["workers"]),
        functional_only=False,
    )
    source = context["paragraphs"]
    # Production bytes are pinned; retain 20 percent headroom before checking the digest.
    maximum = max(len(text.encode()) + 1 for text in source)
    rows = [("production", production, (ledger.contract["machine"]["ram_bytes"], "declared"))]
    rows.append(
        (
            "functional" if cell.get("functional_only") else "production",
            cell,
            resources.require_limit(),
        )
    )
    admitted = True
    for scope, job, limit in rows:
        input_bytes = (
            maximum * job["texts"]
            if job.get("functional_only")
            else int(job["planned_input_bytes"] * 1.2)
        )
        warmup_bytes = input_bytes * min(job["texts"], job["warmup_texts"]) // job["texts"]
        if sizes and job is cell:
            input_bytes, warmup_bytes = sizes
        plan = memory_plan(job, model_bytes, input_bytes, warmup_bytes, work, limit, scope)
        append(
            ledger.root / "memory.jsonl",
            dict(
                cell_id=cell["id"],
                phase="actual" if sizes else "before_input",
                **plan,
            ),
        )
        admitted = admitted and plan["admitted"]
    return admitted


def prepare_input(context, cell, path):
    ledger = context["ledger"]
    source = context["paragraphs"]
    inputs.check_preparation(ledger.admit)
    tokens = headline_tokens(context["models"][cell["model"]]["snapshot"], source)
    identity = inputs.write_input(path, source, [], cell["texts"], 0, tokens, ledger.admit)
    append(ledger.root / "inputs.jsonl", dict(cell_id=cell["id"], **identity))
    atomic(path.with_suffix(".sample.json"), identity["sample_rows"])
    warm = path.with_suffix(".warmup.txt")
    inputs.check_preparation(ledger.admit)
    warm_identity = inputs.write_input(
        warm,
        source,
        [],
        min(cell["texts"], cell["warmup_texts"]),
        0,
        admission=ledger.admit,
    )
    return identity, warm_identity


def input_match(context, cell, identity):
    """The planned production bytes and the writing cell from this invocation must agree."""
    if not cell.get("functional_only") and (
        identity["sha256"] != cell["planned_input_sha256"]
        or identity["file_bytes"] != cell["planned_input_bytes"]
    ):
        return False, "input differs from the fixed production digest or byte count"
    writing = [r for r in context["ledger"].finished if r["id"] == "headline-100m"]
    if writing and writing[0].get("input_identity", {}).get("sha256") != identity["sha256"]:
        return False, "input differs from headline-100m in this invocation"
    return bool(writing), None


def rounds(context, cell, path, identity, warm_identity, launch):
    rows, ledger = [], context["ledger"]
    job = dict(
        cell,
        sample_texts_file=str(path.with_suffix(".sample-texts.json")),
        sample_texts_sha256=sha256(path.with_suffix(".sample-texts.json")),
        warmup_file=str(path.with_suffix(".warmup.txt")),
        warmup_sha256=warm_identity["sha256"],
        warmup_rows=warm_identity["texts"],
    )
    for number in range(cell["rounds"]):
        if not ledger.admit(120):
            break
        rows.append(
            launch(
                context,
                job,
                "fse",
                number,
                path,
                path.with_suffix(".sample.json"),
                identity,
            )
        )
    return rows


def summarize(cell, rows, identity, matches):
    complete = len(rows) == cell["rounds"]
    short = not cell.get("functional_only") and any(r["seconds"] < 2 for r in rows)
    for key in ("identity", "head", "output_sha256"):
        if len({json.dumps(row[key], sort_keys=True) for row in rows}) > 1:
            raise ValueError(f"predict {key} changed between rounds")
    valid = complete and not short
    return dict(
        protocol="predict@1",
        status="invalid" if short else ("valid" if complete else "partial"),
        reason="call shorter than two seconds" if short else (None if complete else "round budget"),
        input_identity=identity,
        input_matches_headline=matches,
        rounds_completed=len(rows),
        sample_checked=bool(rows),
        sample_cpu_bitwise_equal=bool(rows),
        sample_numpy_bitwise_equal=bool(rows),
        label_count=len(LABELS),
        head=rows[0]["head"] if rows else cell["head"],
        output_sha256=rows[0]["output_sha256"] if rows else None,
        output_bytes=rows[0]["output_bytes"] if rows else None,
        seconds=spread([r["seconds"] for r in rows]) if valid else None,
        texts_per_second=spread([cell["texts"] / r["seconds"] for r in rows]) if valid else None,
        publishable=valid
        and not cell.get("functional_only")
        and cell["head"]["kind"] == "artifact",
    )


def run_predict(context, cell, launch, cleanup):
    validate(cell)
    if cell["head"]["kind"] == "placeholder" and not cell.get("functional_only"):
        return dict(status="not_started", reason="publication requires a pinned real toxicity head")
    if not admit_plans(context, cell):
        return dict(status="not_started", reason="predict resource plan rejected before input")
    path = context["work"] / (cell["id"] + ".txt")
    owned = [
        path,
        path.with_suffix(".json"),
        path.with_suffix(".sample.json"),
        path.with_suffix(".sample-texts.json"),
        path.with_suffix(".warmup.txt"),
        path.with_suffix(".warmup.json"),
    ]
    if any(p.exists() or p.is_symlink() for p in owned):
        raise ValueError("predict input paths must be fresh")
    try:
        identity, warm_identity = prepare_input(context, cell, path)
        matches, error = input_match(context, cell, identity)
        if error:
            return dict(
                status="invalid",
                reason=error,
                input_identity=identity,
                seconds=None,
                texts_per_second=None,
                input_matches_headline=False,
            )
        if not admit_plans(context, cell, (identity["file_bytes"], warm_identity["file_bytes"])):
            return dict(status="not_started", reason="predict actual resource plan rejected")
        rows = rounds(context, cell, path, identity, warm_identity, launch)
        return summarize(cell, rows, identity, matches)
    except TimeoutError as error:
        return dict(status="not_started", reason=str(error))
    finally:
        cleanup(context, [p for p in owned if p.exists() or p.is_symlink()])
        sync_directory(context["work"])
        append(
            context["ledger"].root / "cleanup.jsonl",
            dict(
                cell_id=cell["id"],
                input_removed=True,
                completed_at=time.time(),
            ),
        )


def checked_output(output, rows, mode):
    import numpy as np

    dtype = np.bool_ if mode == "labels" else np.float32
    if output.shape != (rows, len(LABELS)) or output.dtype != dtype:
        raise ValueError("predict output shape or dtype differs")
    if mode == "proba":
        for start in range(0, rows, 100000):
            batch = output[start : start + 100000]
            if not np.isfinite(batch).all() or np.any(batch < 0) or np.any(batch > 1):
                raise ValueError("predict probabilities must be finite and in [0, 1]")


def sample_check(job, predict, output):
    import numpy as np

    sample = Path(job["sample_texts_file"])
    if sha256(sample) != job["sample_texts_sha256"]:
        raise ValueError("predict sample texts changed")
    indices = document(Path(job["sample_file"]))
    texts = document(sample)
    if len(indices) != len(texts) or len(indices) != min(10000, job["texts"]):
        raise ValueError("predict sample row count differs")
    actual = output[indices]
    for backend in ("cpu", "numpy"):
        reference = predict(texts, workers=job["workers"], backend=backend)
        checked_output(reference, len(texts), job["output"])
        if not np.array_equal(actual.view(np.uint8), reference.view(np.uint8)):
            raise ValueError(f"predict sample differs bitwise from {backend}")
    return len(indices)


def predict_child(job):
    inputs.configure(job["workers"])
    identity = package_identity("fse")
    for field, digest in (("input_file", "input_sha256"), ("warmup_file", "warmup_sha256")):
        if sha256(Path(job[field])) != job[digest]:
            raise ValueError(f"predict {field} changed")
    model, provenance = classifier(job)
    predict = model.predict if job["output"] == "labels" else model.predict_proba
    warm_started = time.perf_counter()
    warm = predict(Path(job["warmup_file"]), workers=job["workers"])
    warm_seconds = time.perf_counter() - warm_started
    checked_output(warm, job["warmup_rows"], job["output"])
    del warm
    gc.collect()
    if not job.get("functional_only") and warm_seconds < 2:
        raise ValueError("predict warmup shorter than two seconds")
    before = telemetry()
    started = time.perf_counter()
    output = predict(Path(job["input_file"]), workers=job["workers"])
    seconds = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    after = telemetry()
    checked_output(output, job["texts"], job["output"])
    checked_rows = sample_check(job, predict, output)
    if package_identity("fse") != identity:
        raise ValueError("predict executing package identity changed")
    atomic(
        Path(job["receipt"]),
        dict(
            seconds=seconds,
            warmup_seconds=warm_seconds,
            warmup_rows=job["warmup_rows"],
            affinity=sorted(os.sched_getaffinity(0)),
            requested_worker_limit=job["workers"],
            output=job["output"],
            output_bytes=output.nbytes,
            output_sha256=hashlib.sha256(memoryview(output).cast("B")).hexdigest(),
            shape=list(output.shape),
            output_dtype=str(output.dtype),
            head=provenance,
            sample_checked_rows=checked_rows,
            sample_cpu_bitwise_equal=True,
            sample_numpy_bitwise_equal=True,
            peak_rss_bytes=peak,
            identity=identity,
            telemetry_before=before,
            telemetry_after=after,
        ),
    )
