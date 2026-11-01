"""Tiny pod regressions reject false success without loading any model."""

import importlib
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
            "pod4_memory",
            "pod4_receipts",
            "pod_finalize",
            "make_fixup_pod",
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
    assert "timeout --signal=TERM --kill-after=10 900" in script
    assert script.index("quiet-probe") < script.index("run_stage readme-panels")
    assert '"$status" --no-sentinel' in script
    assert '--failure "$export_status" export-failed' in script
    assert "--failure 1 cache-ownership" in script
    assert script.index("DR151_FINALIZER\n") < script.index("if ! /usr/bin/mkdir /dr151-cache")
