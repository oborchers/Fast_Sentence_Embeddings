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
    return {name: importlib.import_module(name) for name in ("pod_duration", "pod_readme_output")}


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
