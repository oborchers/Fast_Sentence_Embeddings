"""Native classifier exports stay bound to the trained files and test split."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def native(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return importlib.import_module("native_classifier")


def test_export_rejects_changed_weights(native, monkeypatch, tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"fitted weights")
    (tmp_path / "training-record.json").write_text(
        json.dumps({"export_files": native.artifact_files(tmp_path)})
    )
    monkeypatch.setitem(
        sys.modules, "model2vec.inference", SimpleNamespace(StaticModelPipeline=object)
    )
    weights.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="differs from its fitted artifact"):
        native.checked_pipeline(tmp_path)


def test_training_sees_only_training_rows_and_evaluates_reload(native, monkeypatch, tmp_path):
    calls = []

    class Pipeline:
        def save_pretrained(self, folder):
            (folder / "model.safetensors").write_bytes(b"trained export")

        @classmethod
        def from_pretrained(cls, folder):
            calls.append(("reload", folder))
            return cls()

        def predict(self, rows, **kwargs):
            calls.append(("predict", list(rows), kwargs))
            return np.array([1, 0])

    class Trainable:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.append(("initialize", kwargs))
            return cls()

        def fit(self, rows, labels, **kwargs):
            calls.append(("fit", list(rows), labels.tolist(), kwargs))

        def to_pipeline(self):
            return Pipeline()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda _: None, Tensor=type("Tensor", (), {}), get_num_threads=lambda: 2
        ),
    )
    monkeypatch.setitem(
        sys.modules, "model2vec.train", SimpleNamespace(StaticModelForClassification=Trainable)
    )
    monkeypatch.setitem(
        sys.modules, "model2vec.inference", SimpleNamespace(StaticModelPipeline=Pipeline)
    )
    monkeypatch.setattr(native, "version", lambda _: "0.10.0")
    monkeypatch.setattr(native, "identity", lambda _: {"runtime_package": "model2vec"})
    monkeypatch.setattr(native, "cached_snapshot", lambda *_: tmp_path)
    monkeypatch.setattr(native, "native_encoder_identity", lambda *_: {"source": "pinned"})
    monkeypatch.setattr(native, "verify_backend_identity", lambda *_: None)
    monkeypatch.setattr(
        native,
        "time",
        SimpleNamespace(monotonic=lambda: pytest.fail("local accuracy must not time training")),
    )
    data = (["train a", "train b"], np.array([0, 1]), ["test a", "test b"], np.array([1, 0]))
    row = native.train(data, "minishlab/potion-base-8M", tmp_path / "export", {})
    fitting = next(call for call in calls if call[0] == "fit")
    assert fitting[1] == data[0]
    assert fitting[3]["test_size"] == 0.1
    assert fitting[3]["random_seed"] == 0
    assert [call[0] for call in calls] == ["initialize", "fit", "reload", "predict"]
    assert row["accuracy"] == 1.0
    assert row["training_seconds"] is None
    assert row["export_files"]
