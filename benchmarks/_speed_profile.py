"""Reconstruct publication inputs offline before profiling an addressed fse child."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from _harness.records import atomic, document, read_rows, sha256
from _measurement import child_record


def verify_artifact(job):
    if sha256(Path(job["artifact"])) != job["artifact_sha256"]:
        raise ValueError("origin artifact hash differs")


def resolve(spec, job, root):
    from _development_profile_origin import single

    if job["backend"] != "fse" or job["kind"] not in ("encode", "dtype", "heads"):
        raise ValueError("speed profiling requires an fse encode, dtype or heads job")
    tag = f"{job['id']}-{job['round']}-{job['label']}"
    if spec["job"] != tag or spec["nominal_arm"] != job["label"]:
        raise ValueError("speed job tag or nominal arm differs")
    row = single(
        r
        for r in read_rows(root, "operations.jsonl")
        if (r["cell_id"], r["round"], r["label"]) == (job["id"], job["round"], job["label"])
    )
    if (
        row["input_sha256"] != job["input_sha256"]
        or "execution" not in row
        or row.get("artifact_sha256") != job.get("artifact_sha256")
    ):
        raise ValueError("speed origin lacks matching input or execution evidence")
    if row["model"] != job["model"]:
        raise ValueError("origin speed model differs")
    verify_artifact(job)
    single(r for r in read_rows(root, "inputs.jsonl") if r["cell_id"] == job["id"])
    return row


def input_child(job):
    from _speed_cells import corpus_sources
    from _speed_run import prepare_input

    os.environ["HF_HUB_OFFLINE"] = "1"
    original = job["original"]
    sentences, unknown, paragraphs = corpus_sources(job["contract"], Path(job["datasets"]))
    context = {
        "paragraphs": sentences,
        "unknown": unknown,
        "paragraph_texts": paragraphs,
        "models": {original["model"]: {"snapshot": original["snapshot"]}},
        "work": Path(job["work"]),
        "ledger": SimpleNamespace(root=Path(job["output"])),
    }
    path, sample, value = prepare_input(context, original)
    if value["sha256"] != original["input_sha256"] or value["sample_rows"] != job["sample_rows"]:
        raise ValueError("rebuilt speed input differs from origin")
    atomic(
        Path(job["receipt"]),
        {
            **child_record(job),
            "input_file": str(path),
            "sample_file": str(sample),
            "identity": value,
        },
    )


def rebuild(spec, origin, job, run_child):
    from _development_profile_origin import single

    root, work = Path(spec["output"]), Path(spec["work"])
    row = single(
        r
        for r in read_rows(Path(spec["source_results"]), "inputs.jsonl")
        if r["cell_id"] == job["id"]
    )
    request = {
        "kind": "profile-input",
        "diagnostic_profile": True,
        "original": origin["job"],
        "contract": origin["environment"]["contract"],
        "datasets": spec["datasets"],
        "work": str(work),
        "output": str(root),
        "sample_rows": row["sample_rows"],
        "receipt": str(root / "rebuilt.json"),
    }
    file = root / "rebuild.job.json"
    atomic(file, request)
    run_child(
        [spec["python"], str(Path(__file__).with_name("speed.py")), "--child-job", str(file)],
        "rebuild",
    )
    rebuilt = document(root / "rebuilt.json")
    job.update(
        input_file=rebuilt["input_file"],
        sample_file=rebuilt["sample_file"],
        vectors_file=str(work / "sample"),
    )


def verify_rebuilt(spec, job):
    from _development_profile_origin import single

    expected = single(
        r
        for r in read_rows(Path(spec["source_results"]), "inputs.jsonl")
        if r["cell_id"] == job["id"]
    )
    if sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("rebuilt speed input changed")
    if document(Path(job["sample_file"])) != expected["sample_rows"]:
        raise ValueError("rebuilt speed sample rows changed")
