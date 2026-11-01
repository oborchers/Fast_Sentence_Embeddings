"""Pod 5 rejects missing scores and failed children, retaining partial evidence."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pod(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    return importlib.import_module("pod5")


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed"},
        {"main_score": float("nan")},
        {"table_dtype": "int8"},
        {"timing_proxy": True},
        {"task": "wrong"},
    ],
)
def test_invalid_score_cannot_complete(pod, tmp_path, change):
    path = tmp_path / "result.jsonl"
    row = {
        "model": "model",
        "task": "task",
        "status": "completed",
        "main_score": 0.5,
        "table_dtype": "float32",
    }
    path.write_text(json.dumps({**row, **change}) + "\n")
    with pytest.raises(ValueError, match="failed, mismatched or nonfinite"):
        pod.verify_task(path, "model", "task")


def test_zero_results_cannot_complete(pod, tmp_path):
    path = tmp_path / "result.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="exactly one"):
        pod.verify_task(path, "model", "task")


def test_child_failure_records_receipt_and_stops(pod, monkeypatch, tmp_path):
    monkeypatch.setattr(pod, "require_space", lambda _: None)
    monkeypatch.setattr(pod.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=17))
    with pytest.raises(RuntimeError, match="exit 17"):
        pod.invoke("task", ["false"], tmp_path, tmp_path)
    assert json.loads((tmp_path / "invocations.jsonl").read_text())["returncode"] == 17


def test_low_disk_stops_before_child(pod, monkeypatch, tmp_path):
    monkeypatch.setattr(pod.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="40 GiB"):
        pod.invoke("task", ["false"], tmp_path, tmp_path)
    assert not (tmp_path / "invocations.jsonl").exists()


@pytest.mark.usefixtures("pod")
def test_shell_and_finalizer_keep_failure_evidence(tmp_path):
    generator = importlib.import_module("make_classics_pod")
    shell = generator.script("a" * 40)
    path = tmp_path / "script.sh"
    path.write_text(shell)
    subprocess.run(["bash", "-n", str(path)], check=True)
    work = shell.split("<<'DR151_WORK'\n", 1)[1].split("DR151_WORK\n", 1)[0]
    path.write_text(work)
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert work.index("run_stage ssh") < work.index("git clone")
    assert "13200 - 120" in shell
    assert "--branch dr151-quality" in work
    assert "150 * 2**30" in work
    assert "sleep infinity" in shell
    assert "shutdown" not in shell
    # Execute only the stdlib finalizer, against a temporary evidence directory.
    finalizer = generator.FINALIZER.replace("Path('/dr151-results')", "Path(sys.argv[1])")
    (tmp_path / "partial.jsonl").write_text('{"task":"x"}\n')
    subprocess.run([sys.executable, "-c", finalizer, str(tmp_path), "124"], check=True)
    receipt = json.loads((tmp_path / "transfer-manifest.json").read_text())
    assert receipt["complete"] is False
    assert receipt["returncode"] == 124
    assert any(
        row["path"] == "partial.jsonl" and len(row["sha256"]) == 64 for row in receipt["files"]
    )


@pytest.mark.usefixtures("pod")
def test_paragram_pin_preserves_earlier_pod_scope():
    sources = importlib.import_module("classic_sources")
    assert len(sources.SOURCES) == 4
    assert sources.ALL_SOURCES["paragram-300-sl999"] == "c99818c1470d5c4e454eb9d59dfce2b179dba455"
