"""The additional folded row and explicit matmul share one raw encoder and linear head."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def preparation(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return importlib.import_module("folded_classifier_prepare")


def test_folded_and_explicit_read_identical_float32_head(preparation, tmp_path):
    sklearn_linear = pytest.importorskip("sklearn.linear_model")

    import fse

    vectors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=np.float32)
    encoder = fse.Encoder.from_arrays(vectors, words=["red", "blue", "green"], normalize=False)
    fitted = sklearn_linear.LogisticRegression()
    fitted.coef_ = np.array([[2, -1, 0.5]], dtype=np.float64)
    fitted.intercept_ = np.array([0.25], dtype=np.float64)
    fitted.classes_ = np.array([0, 1], dtype=np.int64)
    fitted.n_features_in_ = 3
    original, folded = preparation.save_linear(encoder, fitted, tmp_path)
    rows = ["red red", "blue", "green", "unknown", ""]
    with np.load(tmp_path / "linear-head.npz") as head:
        assert head["weight"].dtype == np.float32
        assert np.array_equal(head["weight"], original.coef_)
        assert np.array_equal(head["bias"], original.intercept_)
        explicit = preparation.linear_predict(
            encoder, head["weight"], head["bias"], head["classes"], rows, 1
        )
    assert encoder.steps == ()
    assert folded.standalone
    assert not folded.normalized
    assert folded.n_outputs == 1
    with pytest.raises(ValueError, match="standalone"):
        _ = folded.encoder
    np.testing.assert_array_equal(
        folded.predict(rows, workers=1), original.predict(rows, workers=1)
    )
    np.testing.assert_array_equal(folded.predict(rows, workers=1), explicit)


def test_explicit_path_includes_encoder_then_matmul(preparation):
    calls = []

    def encode(rows, **kwargs):
        calls.append((rows, kwargs))
        return np.array([[2, 0], [0, 3]], dtype=np.float32)

    labels = preparation.linear_predict(
        SimpleNamespace(encode=encode),
        np.array([[1, -1]], np.float32),
        np.array([0], np.float32),
        np.array([10, 20]),
        ["a", "b"],
        32,
    )
    assert calls == [(["a", "b"], {"workers": 32})]
    np.testing.assert_array_equal(labels, [20, 10])


def test_controller_retains_linear_and_mlp_cells_separately(preparation, monkeypatch, tmp_path):
    assert preparation.REPO == "minishlab/potion-base-8M"
    module = importlib.import_module("pod_folded")
    calls = []
    monkeypatch.setattr(
        module, "series", lambda commands, workers, *_: calls.append((commands, workers))
    )
    summary = importlib.import_module("pod_folded_summary")
    monkeypatch.setattr(summary, "publish_summary", lambda *_: None)
    module.timed_dataset(SimpleNamespace(output=tmp_path, deadline=100), tmp_path / "model", "subj")
    module.timed_dataset(
        SimpleNamespace(output=tmp_path, deadline=100), tmp_path / "model", "subj", mlp=True
    )
    assert [workers for _, workers in calls] == [1, 32, 1, 32]
    for commands, _ in calls[:2]:
        assert commands["fse-a"][-1] == commands["fse-b"][-1] == "folded"
        assert commands["encode-matmul"][-1] == "encode-matmul"
        assert commands["model2vec-same-linear"][-1] == "model2vec-linear"
        assert "model2vec-mlp" not in commands
    for commands, _ in calls[2:]:
        assert commands["model2vec-mlp"][-1] == "mlp"


def test_comparator_description_exposes_fine_tuned_mlp(preparation):
    for text in ("fine-tuned", "512-unit ReLU", "PyTorch", "NumPy", "0.10.0"):
        assert text in preparation.MLP_DESCRIPTION


def test_native_model_rules_can_disable_normalization_for_folding(preparation, tmp_path):
    pytest.importorskip("model2vec")
    import fse
    from support.model2vec import save_model

    tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "red", "blue"]
    table = np.arange(len(tokens) * 3, dtype=np.float32).reshape(len(tokens), 3)
    folder = save_model(tmp_path / "model", table=table, tokens=tokens, normalize=True)
    with pytest.raises(ValueError, match="own rules"):
        fse.Encoder.from_pretrained(folder, normalize=False)
    encoder = preparation.raw_mean_encoder(folder, None, local_files_only=False)
    assert encoder.steps == ()
    np.testing.assert_array_equal(
        encoder.encode(["red blue"], workers=1), table[5:7].mean(axis=0, keepdims=True)
    )


@pytest.mark.usefixtures("preparation")
@pytest.mark.parametrize("fails", [False, True])
def test_dataset_failure_cannot_report_success(monkeypatch, tmp_path, fails):
    import sys

    controller = importlib.import_module("pod_folded")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda *_args, **_kwargs: None),
    )
    catalog = tmp_path / "sources.json"
    catalog.write_text('{"datasets": [{"dataset": "SetFit/ag_news"}, {"dataset": "SetFit/subj"}]}')
    monkeypatch.setattr(controller, "SOURCES", catalog)
    monkeypatch.setattr(
        controller,
        "inputs",
        lambda suite, *_: iter([("fixture", (), {})]) if suite == "fineweb" else iter([]),
    )
    monkeypatch.setattr(controller.time, "monotonic", lambda: 0)

    def prepare(*_):
        if fails:
            raise ValueError("fixture preparation failed")

    monkeypatch.setattr(controller, "dataset", prepare)
    args = SimpleNamespace(cache=tmp_path, output=tmp_path, deadline=100, hardware={})
    if fails:
        with pytest.raises(RuntimeError, match="fineweb/fixture"):
            controller.workload(args)
    else:
        controller.workload(args)
    import json

    rows = [json.loads(line) for line in (tmp_path / "invocations.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] == ("failed" if fails else "completed")
