"""Quality reports cannot combine changed runtimes, methods or task revisions."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    return importlib.import_module("protocol")


def test_method_file_changes_alter_the_fingerprint(protocol, tmp_path):
    path = tmp_path / "adapter.py"
    path.write_text("old")
    before = protocol.files_digest(tmp_path, [path])
    path.write_text("new")
    assert protocol.files_digest(tmp_path, [path]) != before


def test_changed_runtime_keeps_partial_scores_in_separate_groups(protocol):
    tasks = [
        SimpleNamespace(
            metadata=SimpleNamespace(
                name=name, model_dump=lambda n=name, **_: {"name": n, "revision": "pinned"}
            )
        )
        for name in ("a", "b")
    ]
    rows = [
        {
            "model": "fse/model",
            "method_fingerprint": "method",
            "runtime_fingerprint": runtime,
            "task": task.metadata.name,
            "task_metadata": task.metadata.model_dump(),
            "main_score": 0.5,
        }
        for runtime, task in zip(("old", "new"), tasks, strict=True)
    ]
    found = protocol.compatible_results(rows, tasks)
    assert len(found) == 2
    assert [len(group) for group in found.values()] == [1, 1]


def test_changed_task_metadata_is_not_accepted(protocol):
    task = SimpleNamespace(
        metadata=SimpleNamespace(name="a", model_dump=lambda **_: {"name": "a", "revision": "new"})
    )
    row = {"model": "m", "task": "a", "task_metadata": {"name": "a", "revision": "old"}}
    assert list(protocol.compatible_results([row], [task]).values()) == [[]]
