"""A fast export must not hide missing or noisy scaling measurements."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("state", ["complete", "missing", "noisy"])
def test_scaling_export_propagates_measurement_coverage(monkeypatch, tmp_path, state):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    module = importlib.import_module("pod_scaling")
    monkeypatch.setattr(module, "emit", lambda *_: None)
    cells = [
        {"cell": f"encode/{model}/float32/wikitext/{workers}/table", "eligible": True}
        for model in (module.POTION, "sentence-transformers/static-retrieval-mrl-en-v1")
        for workers in module.WORKERS
    ]
    if state == "missing":
        cells.pop()
    if state == "noisy":
        cells[-1]["eligible"] = False
    source = tmp_path / "readme-panels"
    source.mkdir()
    (source / "summary.json").write_text(json.dumps({"cells": cells}))
    args = SimpleNamespace(output=tmp_path)
    if state == "complete":
        module.run(args)
    else:
        with pytest.raises(RuntimeError, match="missing or ineligible"):
            module.run(args)
    result = json.loads((tmp_path / "worker-scaling/summary.json").read_text())
    assert result["complete"] == (state == "complete")
    assert result["source"] == str(source / "summary.json")
