"""Large-input pod gates use actual calls and bounded full-output checks."""

import importlib
from pathlib import Path

import numpy as np
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
            "pod_readme_output",
            "pod4_inputs",
            "pod4_readme",
            "make_fixup_pod",
            "pod4_classify",
        )
    }


def test_two_second_gate_cannot_be_passed_by_a_good_median(modules):
    gate = modules["pod_duration"].qualifies
    row = dict(minimum_call_seconds=2, first_encode_s=2.1, warm_seconds=[2.1, 2.2, 2.3])
    assert gate(row)
    assert not gate({**row, "warm_seconds": [1.9, 2.2, 2.3]})
    assert not gate({**row, "first_encode_s": 1.9})
    assert not gate({**row, "warm_seconds": [float("nan")]})
    assert gate({})  # Historical protocols do not acquire a new retrospective gate.


def test_output_validation_covers_later_blocks_and_exact_declared_size(modules):
    contract = modules["pod_readme_output"].contract
    first = np.ones((9000, 2), dtype=np.float32)
    assert contract({"fse": first}, False, 9000)["shape"] == [9000, 2]
    bad = first.copy()
    bad[-1, 0] = 2
    with pytest.raises(ValueError, match="beyond"):
        contract({"fse": first, "reference": bad}, False, 9000)
    with pytest.raises(ValueError, match="9001"):
        contract({"fse": first}, False, 9001)


def test_passage_sampling_is_seeded_with_replacement_and_shuffled(modules):
    sample = modules["pod4_inputs"].sampled_indices
    indices = sample(3, 100, 7)
    np.testing.assert_array_equal(indices, sample(3, 100, 7))
    assert not np.array_equal(indices, sample(3, 100, 8))
    assert len(indices) == 100
    assert len(np.unique(indices)) == 3
    assert all(0 <= index < 3 for index in indices)
    with pytest.raises(ValueError, match="nonempty"):
        sample(0, 100)


def test_slim_plan_has_only_required_settings_and_strict_wrapper(modules, tmp_path):
    panel = modules["pod4_readme"]
    cases = panel.cases()
    assert len(cases) == len(set(cases)) == 21
    assert {case[2] for case in cases} == {1, 16, 32}
    assert sum(case[3] is None for case in cases) == 9
    assert sum(case[3] == "rust" for case in cases) == 3
    script = modules["make_fixup_pod"].script("a" * 40)
    assert "started + 8400 - 90" in script
    assert "100 GB disk" in script
    assert "benchmarks/quality/pod4.py readme" in script
    assert "benchmarks/quality/pod4.py classifications" in script
    for removed in ("large-write-paths", "worker-scaling", "showcase-100m", "mind-evaluate"):
        assert removed not in script
    assert 'if [ "$readme_status" -eq 75 ]; then exit 75; fi' in script
    import subprocess

    path = tmp_path / "pod4.sh"
    path.write_text(script)
    subprocess.run(["/bin/bash", "-n", str(path)], check=True)


def test_classifier_timing_repeats_the_whole_heldout_split(modules, monkeypatch, tmp_path):
    import json
    from collections import Counter
    from types import SimpleNamespace

    module = modules["pod4_classify"]
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    (tmp_path / "pipeline-inputs").mkdir()
    receipt, record = module.expanded_input(tmp_path, ["first", "second"], [0, 1], 5)
    rows = [json.loads(line) for line in (tmp_path / record["file"]).read_text().splitlines()]
    assert len(rows) == record["rows"] == 6
    assert Counter((row["text"], row["label"]) for row in rows) == {
        ("first", 0): 3,
        ("second", 1): 3,
    }
    assert record["source_rows"] == 2
    assert record["repetition_factor"] == 3
    assert receipt.exists()
