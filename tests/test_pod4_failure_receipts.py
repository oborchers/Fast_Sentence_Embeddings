"""Tiny pod regressions reject false success without loading any model."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return {
        name: importlib.import_module(name)
        for name in (
            "pod_duration",
            "pod_supervise",
            "pod4_raw_receipts",
            "pod4_readme",
            "pod4_memory",
            "pod4_receipts",
            "pod_finalize",
            "make_fixup_pod",
            "pod_pairs",
            "pod4_classify",
            "pod_folded_summary",
        )
    }


def test_unfolded_encoder_requires_its_own_long_calls(modules):
    qualifies = modules["pod_duration"].qualifies
    row = dict(
        kind="encode-matmul", minimum_call_seconds=2, first_predict_s=3, warm_seconds=[3, 3, 3]
    )
    assert not qualifies(row)
    assert not qualifies({**row, "encoder_call_seconds": [1, 3, 3, 3]})
    assert qualifies({**row, "encoder_call_seconds": [2, 3, 3, 3]})


def test_memory_admission_includes_current_cgroup_usage(modules, monkeypatch):
    memory = modules["pod4_memory"]
    monkeypatch.setattr(memory, "budget", lambda: (1000, 800))
    assert memory.admit(40, "test")["current"] == 800
    with pytest.raises(MemoryError, match="conservative"):
        memory.admit(60, "test")
    with pytest.raises(ValueError, match="nonnegative integer"):
        memory.admit(float("nan"), "test")


def test_summary_false_or_missing_is_a_failure(modules, monkeypatch, tmp_path):
    receipts = modules["pod4_receipts"]
    assert receipts.check_cells(["expected"], {"cells": [{"cell": "expected", "eligible": False}]})
    assert receipts.check_cells(["expected"], {"cells": []})
    assert not receipts.check_cells(
        ["expected"], {"cells": [{"cell": "expected", "eligible": True}]}
    )
    (tmp_path / "pod4-plan.json").write_text("{}")
    values = []
    finalizer = modules["pod_finalize"]
    monkeypatch.setattr(finalizer, "emit", lambda name, value: values.append((name, value)))
    finalizer.final(tmp_path, 0, sentinel=False)
    manifest = next(value for name, value in values if name == "manifest")
    assert manifest["complete"] is False
    assert manifest["incomplete"]


def test_wrapper_requires_probe_and_exports_setup_failure(modules):
    script = modules["make_fixup_pod"].script("a" * 40)
    assert 'if [ "$probe_status" -ne 0 ]; then exit "$probe_status"; fi' in script
    assert 'if [ "$receipt_status" -ne 0 ]; then exit "$receipt_status"; fi' in script
    assert 'timeout --foreground --signal=TERM --kill-after=10 "$probe_seconds"' in script
    assert script.index("quiet-probe") < script.index("run_stage readme-panels")
    assert '"$status" --no-sentinel' in script
    assert 'failure_manifest "$export_status" export-failed' in script
    assert "failure_manifest 1 cache-ownership" in script
    assert script.index("DR151_FINALIZER\n") < script.index("if ! /usr/bin/mkdir /dr151-cache")


def test_unfolded_short_call_rejects_round(modules, monkeypatch, tmp_path):
    pairs = modules["pod_pairs"]
    controls = dict(
        first_predict_s=3,
        warm_seconds=[3],
        minimum_call_seconds=2,
        input_digest="identical",
        p50_s=3,
        runtime_package="fse",
    )
    rows = {
        "fse-a": {**controls, "label": "fse-a"},
        "fse-b": {**controls, "label": "fse-b"},
        "encode-matmul": {
            **controls,
            "label": "encode-matmul",
            "kind": "encode-matmul",
            "encoder_call_seconds": [1, 1],
        },
    }
    monkeypatch.setattr(pairs, "operation", lambda _command, label, *_args: rows[label])
    good, _ = pairs.round_observation(rows, list(rows), 1, 100, tmp_path, {}, {})
    assert good is False


def test_completed_series_cannot_hide_ineligible_classifier(modules, monkeypatch, tmp_path):
    from types import SimpleNamespace

    classify = modules["pod4_classify"]
    summary = modules["pod_folded_summary"]
    monkeypatch.setattr(classify, "calibrate", lambda *_args: (tmp_path / "input.json", {}))
    monkeypatch.setattr(classify, "series", lambda *_args: {"complete": True})
    monkeypatch.setattr(summary, "publish_summary", lambda *_args: {"cells": []})
    args = SimpleNamespace(output=tmp_path, deadline=100)
    with pytest.raises(RuntimeError, match="short, failed or noisy"):
        classify.timed(args, tmp_path, ["text"], [0], "setfit", "test")


def test_export_failure_mode_emits_decodable_failed_manifest(modules):
    import base64
    import gzip
    import subprocess
    import sys

    path = Path(modules["pod_finalize"].__file__)
    done = subprocess.run(
        [sys.executable, str(path), "--failure", "124", "export-failed"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = "".join(
        line.split()[3]
        for line in done.stdout.splitlines()
        if line.startswith("DR190_DATA manifest ")
    )
    value = json.loads(gzip.decompress(base64.b64decode(payload)))
    assert value["complete"] is False
    assert value["returncode"] == 124
    assert "ALL_DONE" not in done.stdout


def test_nested_first_call_cannot_replace_a_missing_warm_call(modules):
    assert not modules["pod_duration"].qualifies(
        dict(
            kind="encode-matmul",
            minimum_call_seconds=2,
            first_predict_s=3,
            encoder_call_seconds=[3],
            warm_seconds=[],
        )
    )


def test_fresh_cache_prepares_model_before_input_statistics(modules, monkeypatch):
    from types import SimpleNamespace

    module = modules["pod4_readme"]
    order = []
    monkeypatch.setattr(module, "options_for", lambda *_: SimpleNamespace())
    monkeypatch.setattr(module, "ensure_model", lambda *_: order.append("model"))

    def stop(_cache):
        order.append("source")
        raise RuntimeError("stop before any work")

    monkeypatch.setattr(module, "source_passages", stop)
    with pytest.raises(RuntimeError, match="stop before any work"):
        module.probe(SimpleNamespace(cache=None))
    assert order == ["model", "source"]


def test_missing_raw_cell_is_rejected_despite_eligible_summary(modules, tmp_path):
    (tmp_path / "speed-results.jsonl").write_text(
        json.dumps({"event": "operation", "cell": "expected", "label": "fse-a"}) + "\n"
    )
    (tmp_path / "output-checks.jsonl").write_text("")
    result = modules["pod4_raw_receipts"].readme(tmp_path, ["expected"])
    assert result["cells"] == [{"cell": "expected", "eligible": False}]


def test_workload_supervisor_returns_status_and_enforces_deadline(modules, tmp_path):
    import sys

    run = modules["pod_supervise"].run
    assert run([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "exit.log", 5) == 7
    assert (
        run([sys.executable, "-c", "import signal; signal.pause()"], tmp_path / "timeout.log", 0.1)
        == 124
    )
