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
    monkeypatch.setenv("DR151_DEADLINE", "9999999999")
    monkeypatch.setattr(pod, "memory_snapshot", lambda: {})
    monkeypatch.setattr(pod, "run_bounded", lambda *_a: 17)
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
    assert "12900 - 120" in shell
    assert "--branch dr151-quality" in work
    assert "150 * 2**30" in work
    assert "sleep infinity" in shell
    assert "shutdown" not in shell
    # Execute only the stdlib finalizer, against a temporary evidence directory.
    finalizer = generator.FINALIZER.replace("Path('/dr151-results')", "Path(sys.argv[1])")
    model_cache = tmp_path / "models"
    model_cache.mkdir()
    (model_cache / "example-fit.npz").write_bytes(b"fitted state")
    finalizer = finalizer.replace("Path('/dr151-cache/models')", "Path(sys.argv[1]) / 'models'")
    (tmp_path / "partial.jsonl").write_text('{"task":"x"}\n')
    subprocess.run([sys.executable, "-c", finalizer, str(tmp_path), "124"], check=True)
    receipt = json.loads((tmp_path / "transfer-manifest.json").read_text())
    assert (tmp_path / "example-fit.npz").read_bytes() == b"fitted state"
    assert isinstance(json.loads((tmp_path / "container-memory.json").read_text()), dict)
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


def test_timeout_reaps_child_group_and_reports_failure(pod, monkeypatch):
    signals = []

    class Child:
        pid = 123
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            return 0

    monkeypatch.setattr(pod.subprocess, "Popen", lambda *_a, **_k: Child())
    monkeypatch.setattr(pod.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    assert pod.run_bounded(["child"], 1) == 124
    assert signals == [(123, pod.signal.SIGTERM), (123, pod.signal.SIGKILL)]


@pytest.fixture
def memory_tree(pod, tmp_path):
    memory = importlib.import_module(pod.memory_snapshot.__module__)
    root, proc = tmp_path / "cgroup", tmp_path / "proc"
    root.mkdir()
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/\n")
    (proc / "self/mountinfo").write_text(f"1 0 0:1 /container {root} ro - cgroup2 cgroup rw\n")
    for name, value in {
        "memory.max": 64 * 2**30,
        "memory.current": 2**30,
        "memory.peak": 2 * 2**30,
        "cgroup.procs": 42,
    }.items():
        (root / name).write_text(str(value) + "\n")
    return memory, root, proc


def test_memory_counters_bind_to_current_container(memory_tree):
    memory, root, proc = memory_tree
    state = memory.snapshot(root, proc, 42)
    assert state["memory.max"] == 64 * 2**30
    assert state["memory.peak"] == 2 * 2**30


@pytest.mark.parametrize(
    ("file", "value"), [("memory.max", "max"), ("cgroup.procs", "777"), ("memory.peak", "0")]
)
def test_unproven_memory_cannot_pass(memory_tree, file, value):
    memory, root, proc = memory_tree
    (root / file).write_text(value + "\n")
    with pytest.raises(RuntimeError):
        memory.snapshot(root, proc, 42)


def test_nested_cgroup_cannot_ignore_ancestor_limits(memory_tree):
    memory, root, proc = memory_tree
    (proc / "self/cgroup").write_text("0::/nested\n")
    with pytest.raises(RuntimeError, match="private cgroup"):
        memory.snapshot(root, proc, 42)


def test_missing_peak_counter_cannot_pass(memory_tree):
    memory, root, proc = memory_tree
    (root / "memory.peak").rename(root / "unavailable-counter")
    with pytest.raises(FileNotFoundError):
        memory.snapshot(root, proc, 42)


def test_cleanup_refuses_cache_without_ownership(pod, tmp_path):
    with pytest.raises(ValueError, match="exclusively created"):
        pod.discard_cache(tmp_path / "intermediate", tmp_path, tmp_path)


def test_preflight_rejects_low_namespace_ram(memory_tree, monkeypatch):
    memory, _, _ = memory_tree
    monkeypatch.setattr(memory, "snapshot", lambda: {"memory.max": 16 * 2**30, "memory.current": 0})
    with pytest.raises(RuntimeError, match="58 GiB"):
        memory.preflight()
