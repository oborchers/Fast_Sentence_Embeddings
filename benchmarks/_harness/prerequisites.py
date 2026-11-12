"""Admission receipts binding successful steps to one allocation and its inputs."""

from __future__ import annotations

from pathlib import Path

from _harness.collect import verify
from _harness.records import read_rows, sha256


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
        "parameters": values,
        "files": {str(p.resolve()): sha256(p) for p in sorted(files)},
    }


def receipt(results, context):
    """Only a verified successful seal can supply a prerequisite receipt."""
    _, done = verify(results)
    if type(done["exit_status"]) is not int or done["exit_status"] != 0:
        raise ValueError("prerequisite DONE exit status is not zero")
    return {"identity": context, "done_sha256": sha256(results / "DONE")}


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
