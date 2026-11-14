"""Single-call native-file classification headlines, with owned input and memory plans.

The placeholder is functional evidence only. Publication needs a pinned real head of kind
artifact, loaded offline from its pinned snapshot and checked against its SHA-256. No model is
trained by this module.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import resource
import time
from pathlib import Path

import _speed_cells as inputs
from _harness import resources
from _harness.pressure import interval, report
from _harness.records import append, atomic, document, sha256, sync_directory
from _measurement import package_identity, spread, telemetry
from _measurement_config import MODEL_PINS
from _speed_write import headline_tokens

PROFILE = "toxicity-placeholder@1"
ARTIFACT_KEYS = ("kind", "repo", "revision", "filename", "sha256", "labels")


def _validate_labels(labels):
    if (
        not isinstance(labels, list)
        or not 1 <= len(labels) <= 16
        or not all(isinstance(label, str) and label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("head labels must be 1 to 16 unique nonempty strings")


def validate_artifact(head):
    """Check the pinned real head in its original form before anything is downloaded."""
    if set(head) != set(ARTIFACT_KEYS):
        raise ValueError(
            "artifact head requires exactly kind, repo, revision, filename, sha256, labels"
        )
    for key in ("repo", "revision", "filename", "sha256"):
        if not isinstance(head[key], str):
            raise ValueError(f"artifact head {key} must be a string")
    if not re.fullmatch(r"[0-9A-Za-z][\w.-]*/[0-9A-Za-z][\w.-]*", head["repo"]):
        raise ValueError("artifact head repo must be namespace/name")
    if not re.fullmatch(r"[0-9a-f]{40}", head["revision"]):
        raise ValueError("artifact head revision must be a full commit hash")
    if not re.fullmatch(r"[\w][\w.-]*\.safetensors", head["filename"]):
        raise ValueError("artifact head filename must be a plain .safetensors file name")
    if not re.fullmatch(r"[0-9a-f]{64}", head["sha256"]):
        raise ValueError("artifact head sha256 must be a SHA256 hex string")
    if MODEL_PINS.get(head["repo"]) != head["revision"]:
        raise ValueError("artifact head repo and revision must match pins.json")


def validate(cell):
    head = cell["head"]
    if cell["output"] not in ("labels", "proba"):
        raise ValueError("predict output must be labels or proba")
    _validate_labels(head.get("labels"))
    if head["kind"] not in ("placeholder", "artifact"):
        raise ValueError("head kind must be placeholder or artifact")
    if head["kind"] == "placeholder" and (head["profile"] != PROFILE or head["seed"] != 17):
        raise ValueError("placeholder requires toxicity-placeholder@1 and seed 17")
    if head["kind"] == "artifact":
        validate_artifact(head)
    if cell["rounds"] != 3 or cell["warmup_texts"] <= 0:
        raise ValueError("predict requires three rounds and positive warmup_texts")


def artifact_classifier(job):
    """The pinned real head, loaded offline from its snapshot and checked against the cell."""
    from fse import Classifier

    head = job["head"]
    snapshot = Path(job["head_snapshot"])
    if snapshot.name != head["revision"]:
        raise ValueError("head snapshot is not the pinned revision")
    weights = snapshot / head["filename"]
    if sha256(weights) != head["sha256"]:
        raise ValueError("head file differs from the pinned SHA-256")
    labels = tuple(head["labels"])
    model = Classifier.from_pretrained(snapshot, mmap=True)
    if model.path() != "standalone" or tuple(model.classes_) != labels:
        raise ValueError("headline head must be the standalone folded head with ordered labels")
    provenance = dict(
        head,
        link=model.link,
        normalized=model.normalized,
        path=model.path(),
        thresholds=dict(model.thresholds),
        fingerprint=model.fingerprint,
        note="Pinned published toxicity classifier; no training",
    )
    return model, provenance


def classifier(job):
    import numpy as np

    from fse import Classifier, Encoder

    validate(job)
    if job["head"]["kind"] == "artifact":
        return artifact_classifier(job)
    if not job.get("functional_only"):
        raise ValueError("placeholder heads are only allowed in functional probes")
    encoder = Encoder.from_pretrained(job["artifact"], normalize=False, mmap=True)
    labels = tuple(job["head"]["labels"])
    rng = np.random.default_rng(job["head"]["seed"])
    weight = rng.standard_normal((len(labels), encoder.output_dim)).astype(np.float32)
    bias = rng.standard_normal(len(labels)).astype(np.float32)
    model = Classifier.from_arrays(
        encoder,
        weight,
        bias,
        labels=labels,
        link="multilabel",
        normalized=False,
    )
    if model.path() != "folded" or tuple(model.classes_) != labels:
        raise ValueError("headline head must be folded with the ordered labels")
    if dict(model.thresholds) != dict.fromkeys(labels, 0.5):
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


def memory_plan(
    cell,
    model_bytes,
    input_bytes,
    warmup_bytes,
    work,
    limit,
    kind,
    occupied_disk=0,
):
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
        disk=max(0, input_bytes + warmup_bytes - occupied_disk) + 5 * resources.GIB,
        occupied_input_disk_bytes=occupied_disk,
        kind=kind,
        output=cell["output"],
        label_count=k,
    )


def head_bytes(context, cell):
    """Bytes the child maps: the pinned head file, or the encoder artifact for a placeholder."""
    head = cell["head"]
    if head["kind"] == "artifact":
        return (Path(context["models"][head["repo"]]["snapshot"]) / head["filename"]).stat().st_size
    return Path(context["models"][cell["model"]]["artifact"]).stat().st_size


def admit_plans(context, cell, sizes=None):
    """Record production and actual plans before input, then again with known file sizes."""
    ledger, work = context["ledger"], context["work"]
    model_bytes = head_bytes(context, cell)
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
        plan = memory_plan(
            job,
            model_bytes,
            input_bytes,
            warmup_bytes,
            work,
            limit,
            scope,
            occupied_disk=sum(sizes) if sizes else 0,
        )
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
    head = cell["head"]
    job = dict(
        cell,
        head_snapshot=(
            context["models"][head["repo"]]["snapshot"] if head["kind"] == "artifact" else None
        ),
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
        status="invalid"
        if short
        else ("valid" if complete else ("partial" if rows else "not_started")),
        reason="call shorter than two seconds" if short else (None if complete else "round budget"),
        input_identity=identity,
        input_matches_headline=matches,
        cpu_pressure=report(rows),
        rounds_completed=len(rows),
        sample_checked=bool(rows),
        sample_cpu_bitwise_equal=bool(rows),
        sample_numpy_bitwise_equal=bool(rows),
        label_count=len(cell["head"]["labels"]),
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


def checked_output(output, rows, mode, label_count):
    import numpy as np

    dtype = np.bool_ if mode == "labels" else np.float32
    if output.shape != (rows, label_count) or output.dtype != dtype:
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
        checked_output(reference, len(texts), job["output"], len(job["head"]["labels"]))
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
    warm_before = telemetry()
    warm_started = time.perf_counter()
    warm = predict(Path(job["warmup_file"]), workers=job["workers"])
    warm_seconds = time.perf_counter() - warm_started
    warm_after = telemetry()
    checked_output(warm, job["warmup_rows"], job["output"], len(job["head"]["labels"]))
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
    checked_output(output, job["texts"], job["output"], len(job["head"]["labels"]))
    checked_rows = sample_check(job, predict, output)
    if package_identity("fse") != identity:
        raise ValueError("predict executing package identity changed")
    atomic(
        Path(job["receipt"]),
        dict(
            seconds=seconds,
            warmup_seconds=warm_seconds,
            warmup_cpu_pressure=interval(warm_before, warm_after),
            warmup_telemetry_before=warm_before,
            warmup_telemetry_after=warm_after,
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
            cpu_pressure=interval(before, after),
        ),
    )
