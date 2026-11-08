"""Resolve exactly one sealed child and preserve its content provenance."""

from __future__ import annotations

from pathlib import Path

from _harness.collect import verify
from _harness.profile import identity, sources, verify_inputs
from _harness.records import document, read_rows, sha256

HERE = Path(__file__).resolve().parent


def single(rows):
    rows = list(rows)
    if len(rows) != 1:
        raise ValueError("exactly one matching origin record required")
    return rows[0]


def resolve(spec):
    root = Path(spec["source_results"])
    members, _ = verify(root)
    tag = spec["job"]
    if tag + ".job.json" not in members:
        raise ValueError("addressed job is absent from sealed origin")
    file = root / (tag + ".job.json")
    job = document(file)
    environment = document(root / "environment.json")
    if environment["measurement_source_sha256"] != sources(HERE):
        raise ValueError("measurement sources differ from origin")
    if spec["entry"] == "development":
        row = _development(spec, job, root, members)
        verify_inputs(job)
    else:
        from _speed_profile import resolve as speed_origin

        row = speed_origin(spec, job, root)
    _work_paths(spec, job)
    return {
        "job": job,
        "row": row,
        "identity": identity(row, spec["entry"]),
        "job_sha256": sha256(file),
        "environment": environment,
    }


def _development(spec, job, root, members):
    prefix = job["id"] + "-"
    if not spec["job"].startswith(prefix):
        raise ValueError("job tag does not match its cell")
    phase, pair, position = spec["job"][len(prefix) :].rsplit("-", 2)
    row = single(
        r
        for r in read_rows(root, "operations.jsonl")
        if (r["cell_id"], r["phase"], r["pair"], r["position"])
        == (job["id"], phase, int(pair), int(position))
    )
    receipt_name = spec["job"] + ".json"
    if receipt_name not in members:
        raise ValueError("origin receipt is absent from sealed inventory")
    receipt = document(root / receipt_name)
    if any(row.get(key) != value for key, value in receipt.items()):
        raise ValueError("origin receipt and operation differ")
    for key in ("texts", "profile"):
        if row[key] != job[key]:
            raise ValueError("origin job and operation workload differ")
    _workload(job, receipt)
    if row["requested_workers"] != job["workers"]:
        raise ValueError("origin worker count differs")
    if row["arm"] != spec["nominal_arm"] or job["action"] != "measure":
        raise ValueError("nominal arm or origin action differs")
    if job["tag"] != spec["job"]:
        raise ValueError("origin tag differs")
    return row


def _workload(job, receipt):
    if any(receipt["input"][key] != job["corpus"][key] for key in ("path", "sha256")):
        raise ValueError("origin corpus differs between job and receipt")
    if receipt["operation_repetitions"] != job["iterations"]:
        raise ValueError("origin iteration count differs between job and receipt")
    if receipt["heads_mode"] != job.get("heads_mode"):
        raise ValueError("origin frozen heads mode differs between job and receipt")


def check_work_paths(spec):
    file = Path(spec["source_results"]) / (spec["job"] + ".job.json")
    if file.is_file():
        _work_paths(spec, document(file))


def _work_paths(spec, job):
    work = Path(job["work"]) if "work" in job else Path(job["vectors_file"]).parent.parent
    protected = [work.resolve()]
    if spec["datasets"] is not None:
        protected.append(Path(spec["datasets"]).resolve())
    for key in ("output", "work"):
        path = Path(spec[key]).resolve()
        if any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
            raise ValueError("profile folders overlap origin work or datasets")


def verify_again(spec, origin):
    verify(Path(spec["source_results"]))
    if origin["environment"]["measurement_source_sha256"] != sources(HERE):
        raise ValueError("measurement sources changed during profiling")
    if spec["entry"] == "development":
        verify_inputs(origin["job"])
    else:
        from _speed_profile import verify_artifact

        verify_artifact(origin["job"])
