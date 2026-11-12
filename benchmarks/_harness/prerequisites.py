"""Admission receipts binding successful steps to one allocation and its inputs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from _harness.collect import verify
from _harness.records import document, read_rows, seal, sha256


def runner_identity():
    """Identify the executing runner revision, including dirty or added Python sources."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("executing runner revision could not be read") from error
    return {
        "revision": revision,
        "sources": {
            str(p.relative_to(root)): sha256(p) for p in sorted((root / "benchmarks").rglob("*.py"))
        },
    }


def identity(allocation, values):
    """Hash current locks, revisions and configuration, never just their path names."""
    source = Path(values.get("source", Path(__file__).resolve().parents[2]))
    config = source / "benchmarks/config"
    files = set(config.glob("*.json"))
    files.update(config / e["lock"] for e in allocation.definition["environments"])
    if "input" in values:
        files.add(Path(values["input"]))
    setup = allocation.root / f"{allocation.entry}-env/setup.json"
    if setup.exists():
        files.add(setup)
    return {
        "root": str(allocation.root.resolve()),
        "entry": allocation.entry,
        "allocated_at": allocation.allocated_at,
        "definition": allocation.definition,
        "runner": runner_identity(),
        "parameters": values,
        "files": {str(p.resolve()): sha256(p) for p in sorted(files)},
    }


def receipt(results, context):
    """Only a verified successful seal can supply a prerequisite receipt."""
    _, done = verify(results)
    if type(done["exit_status"]) is not int or done["exit_status"] != 0:
        raise ValueError("prerequisite DONE exit status is not zero")
    return {"identity": context, "done_sha256": sha256(results / "DONE")}


def completion_status(results, code):
    """Malformed completion evidence fails the step without hiding its dependents."""
    try:
        status = document(results / "DONE")["exit_status"]
        if type(status) is not int:
            raise ValueError("DONE exit status must be an integer")
        return status or code, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return code or 1, f"invalid DONE: {error}"


def snapshot(record):
    """Called only by Allocation before starting decide, never rewrite a snapshot."""
    target = record / "decision-inputs"
    target.mkdir(exist_ok=False)
    for name in ("allocation.json", "machine-check.json", "steps.jsonl"):
        shutil.copyfile(record / name, target / name)
    seal(target, 0, 0, extra={"completed_at": None})


def recorded(root, names):
    """Verify saved allocation receipts without reopening relocated work or environments.

    The original identity remains content, including its original absolute paths.
    All steps must name that same identity and the saved allocation definition.
    This uses the admission receipt verifier, also after collection.
    """
    record = root / "allocation-results"
    if (record / "DONE").exists():
        verify(record)
    snapshot_root = record / "decision-inputs"
    if snapshot_root.exists() or snapshot_root.is_symlink():
        record = snapshot_root
    verify(record)
    saved = document(record / "allocation.json")
    checked = document(record / "machine-check.json")
    if checked["status"] != "accepted" or checked["rehearsal"] is not False:
        raise ValueError("accepted production allocation required")
    rows = read_rows(record, "steps.jsonl")
    context, receipts = None, {}
    declared = [s["name"] for s in saved["definition"]["steps"]]
    for name in names:
        matches = [row for row in rows if row["step"] == name]
        if declared.count(name) != 1 or len(matches) != 1:
            raise ValueError("required allocation step is missing or duplicated")
        row = matches[0]
        if row["status"] != "finished" or type(row["exit_status"]) is not int or row["exit_status"]:
            raise ValueError("required allocation step did not finish successfully")
        evidence = row["evidence"]
        current = evidence["identity"]
        if context is None:
            context = current
        if current != context or current["entry"] != "speed":
            raise ValueError("required steps belong to different allocations")
        if any(current[key] != saved[key] for key in ("definition", "allocated_at")):
            raise ValueError("step identity differs from saved allocation")
        folder = name + "-results"
        if row["results"] != folder or receipt(root / folder, context) != evidence:
            raise ValueError("required step seal differs from its admission receipt")
        receipts[name] = evidence["done_sha256"]
    return context, receipts


def refusal(allocation, step, values):
    """Return an explicit refusal, or None when every prerequisite is satisfied."""
    requires = step.get("requires", [])
    if not requires:
        return None
    try:
        context = identity(allocation, values)
        rows = read_rows(allocation.record, "steps.jsonl")
        for name in requires:
            matches = [row for row in rows if row["step"] == name]
            if len(matches) != 1 or matches[0]["status"] != "finished":
                return f"prerequisite {name} did not finish successfully"
            row = matches[0]
            if type(row["exit_status"]) is not int or row["exit_status"] != 0:
                return f"prerequisite {name} failed"
            results = allocation.results({"name": name})
            if row.get("evidence") != receipt(results, context):
                return f"prerequisite {name} evidence does not match this allocation"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f"prerequisite evidence unavailable or invalid: {error}"
    return None


def deadline_refusal(allocation, step):
    deadline = step.get("requires_by_seconds")
    if deadline is not None and allocation.budget.elapsed() >= deadline:
        return "prerequisite admission deadline reached"
    return None
