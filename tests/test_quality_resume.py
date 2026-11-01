"""A changed evaluator, encoder or dataset cannot reuse an MTEB result."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/quality/resume.py"
SPEC = importlib.util.spec_from_file_location("quality_resume", PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs():
    metadata = {"name": "task", "dataset": {"revision": "pinned"}}
    task = SimpleNamespace(metadata=SimpleNamespace(name="task", model_dump=lambda **_: metadata))
    specification = dict(
        zip(
            MODULE.KEYS,
            ("fse/model", "rev", "loader", "fingerprint", 2, "cpu", {"mteb": "pinned"}),
            strict=True,
        )
    )
    row = {**specification, "task": "task", "task_metadata": metadata, "main_score": 0.5}
    return task, specification, row


def test_resume_only_reuses_matching_complete_task(tmp_path):
    task, specification, row = inputs()
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert MODULE.pending_tasks(path, specification, [task]) == []


@pytest.mark.parametrize("field", MODULE.KEYS)
def test_changed_measurement_identity_requires_rerun(tmp_path, field):
    task, specification, row = inputs()
    row[field] = "changed"
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert MODULE.pending_tasks(path, specification, [task]) == [task]


def test_changed_dataset_revision_requires_rerun(tmp_path):
    task, specification, row = inputs()
    row["task_metadata"] = {"name": "task", "dataset": {"revision": "other"}}
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert MODULE.pending_tasks(path, specification, [task]) == [task]
