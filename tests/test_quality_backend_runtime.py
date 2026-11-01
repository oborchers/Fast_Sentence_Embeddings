"""Quality rows identify actual model code separately from their comparison context."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    return importlib.import_module("backend_runtime")


@pytest.mark.parametrize(
    ("package", "kind"),
    [("fse", "fse"), ("model2vec", "model2vec"), ("sentence_transformers", "st")],
)
def test_actual_inner_model_selects_its_own_runtime(backend, monkeypatch, package, kind):
    model_type = type("Encoder", (), {"__module__": package + ".implementation"})
    calls = []

    def identity(actual):
        calls.append(actual)
        return {"runtime_fingerprint": actual + "-code", "package_location": "/" + actual}

    monkeypatch.setattr(backend, "backend_identity", identity)
    row = backend.model_runtime(
        SimpleNamespace(model=model_type()),
        {"fse_reference_runtime_fingerprint": "common-reference"},
    )
    assert calls == [kind]
    assert row["runtime_fingerprint"] == kind + "-code"
    assert row["package_location"] == "/" + kind
    assert row["fse_reference_runtime_fingerprint"] == "common-reference"


def test_unknown_model_code_cannot_be_labeled_as_fse(backend):
    with pytest.raises(ValueError, match="cannot identify"):
        backend.model_runtime(
            SimpleNamespace(model=object()), {"fse_reference_runtime_fingerprint": "reference"}
        )
