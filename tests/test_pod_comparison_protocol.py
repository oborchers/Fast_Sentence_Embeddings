"""Pod validity uses A/A and immutable complete inputs, never shared-host load."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def pairs(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return importlib.import_module("pod_pairs")


@pytest.mark.parametrize(("ratio", "accepted"), [(1.09, True), (1.11, False)])
def test_host_load_does_not_replace_aa_gate(pairs, monkeypatch, tmp_path, ratio, accepted):
    def operation(_command, label, *_args):
        return {
            "label": label,
            "input_digest": "full corpus",
            "p50_s": ratio if label == "fse-b" else 1.0,
            "load_before": [59, 59, 59],
            "load_after": [59, 59, 59],
        }

    monkeypatch.setattr(pairs, "operation", operation)
    valid, observed = pairs.round_observation(
        dict.fromkeys(["fse-a", "fse-b", "model2vec"], ()),
        ["fse-a", "fse-b", "model2vec"],
        32,
        100,
        tmp_path,
        {},
        {},
    )
    assert valid is accepted
    assert observed["aa_max_ratio"] == ratio
    rows = [
        json.loads(line) for line in (tmp_path / "speed-results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert rows[-1]["symmetric_aa_band"] == [1 / ratio, ratio]


def test_different_complete_inputs_never_form_valid_pair(pairs, monkeypatch, tmp_path):
    monkeypatch.setattr(
        pairs,
        "operation",
        lambda _command, label, *_: {
            "label": label,
            "p50_s": 1.0,
            "input_digest": "changed" if label == "model2vec" else "original",
        },
    )
    valid, _ = pairs.round_observation(
        dict.fromkeys(["fse-a", "fse-b", "model2vec"], ()),
        ["fse-a", "fse-b", "model2vec"],
        1,
        100,
        tmp_path,
        {},
        {},
    )
    assert not valid


def test_rotates_all_contenders_and_retains_noisy_rounds(pairs, monkeypatch, tmp_path):
    monkeypatch.setattr(pairs.sys, "platform", "linux")
    monkeypatch.setattr(pairs, "environment", lambda: {})
    calls = []

    def observed(_commands, ordered, *_):
        calls.append(ordered)
        return (
            (False, {"aa_max_ratio": 1.2, "control_seconds": [1, 1.2]})
            if len(calls) == 1
            else (True, {"aa_max_ratio": 1.01, "control_seconds": [1, 1.01]})
        )

    monkeypatch.setattr(pairs, "round_observation", observed)
    result = pairs.series(
        {"fse-a": [], "fse-b": [], "model2vec": []}, 1, tmp_path, float("inf"), "cell"
    )
    assert calls[:3] == [
        ["fse-a", "fse-b", "model2vec"],
        ["fse-b", "model2vec", "fse-a"],
        ["model2vec", "fse-a", "fse-b"],
    ]
    assert result["valid_rounds"] == 3
    assert result["aa_max_ratio"] == 1.2
    assert not result["complete"]


def test_between_round_drift_rejects_cell(pairs, monkeypatch, tmp_path):
    monkeypatch.setattr(pairs.sys, "platform", "linux")
    monkeypatch.setattr(pairs, "environment", lambda: {})
    times = iter([1.0, 2.0, 3.0])

    def observed(*_):
        seconds = next(times)
        return True, {"aa_max_ratio": 1.0, "control_seconds": [seconds, seconds]}

    monkeypatch.setattr(pairs, "round_observation", observed)
    result = pairs.series({"fse-a": [], "fse-b": []}, 1, tmp_path, float("inf"), "cell")
    assert result["valid_rounds"] == 3
    assert result["aa_within_round_max_ratio"] == 1
    assert result["aa_max_ratio"] == 3
    assert result["aa_control_max_ratio"] == 3
    assert not result["complete"]


def test_short_probe_cannot_pass_with_three_good_rounds(pairs, monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(pairs.sys, "platform", "linux")
    monkeypatch.setattr(pairs, "environment", lambda: {})
    moments = iter([0, 1, 2, 3, 4, 11, 12])
    monkeypatch.setattr(pairs, "time", SimpleNamespace(monotonic=lambda: next(moments)))
    monkeypatch.setattr(
        pairs,
        "round_observation",
        lambda *_: (True, {"aa_max_ratio": 1, "control_seconds": [1, 1]}),
    )
    result = pairs.series({"fse-a": [], "fse-b": []}, 1, tmp_path, 10, "probe", minimum_seconds=120)
    assert result["valid_rounds"] == 3
    assert result["seconds"] == 12
    assert not result["complete"]
