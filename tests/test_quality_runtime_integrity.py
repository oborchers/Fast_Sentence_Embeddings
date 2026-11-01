"""Reject changed executing code before accepting classifier or search evidence."""

import importlib
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    monkeypatch.syspath_prepend(str(root / "benchmarks/quality"))
    return {name: importlib.import_module(name) for name in ("compare_runtime", "protocol")}


def test_quality_reference_is_rechecked_instead_of_reusing_a_stale_hash(modules, monkeypatch):
    protocol = modules["protocol"]
    monkeypatch.setattr(protocol, "runtime_identity", lambda: "before")
    reference = {"fse_reference_runtime_fingerprint": "before"}
    protocol.verify_reference_runtime(reference)
    monkeypatch.setattr(protocol, "runtime_identity", lambda: "after")
    with pytest.raises(ValueError, match="reference runtime changed"):
        protocol.verify_reference_runtime(reference)


@pytest.mark.parametrize(
    ("module_name", "kind"),
    [("prepare_fineweb", "model2vec"), ("prepare_setfit", "tfidf")],
)
def test_classifier_cannot_append_after_code_changes_during_prediction(
    modules, monkeypatch, tmp_path, module_name, kind
):
    module = importlib.import_module(module_name)
    runtime = modules["compare_runtime"]
    current = {"hash": "before-prediction"}

    def identity(_):
        return {
            "runtime_package": runtime.expected_package(kind),
            "runtime_package_version": "pinned",
            "package_location": "/backend/code",
            "runtime_fingerprint": current["hash"],
        }

    monkeypatch.setattr(runtime, "backend_identity", identity)
    monkeypatch.setattr(module, "backend_identity", identity)
    before = module.backend_identity(kind)
    args = SimpleNamespace(results=tmp_path / "scores.jsonl")
    current["hash"] = "changed-during-prediction"
    if module_name == "prepare_fineweb":
        operation = partial(
            module.record, args, {**before, "backend": kind, "model": "test", "macro_f1": 1}
        )
    else:
        operation = partial(module.save_result, args, before, kind, "test", [0], [0])
    with pytest.raises(ValueError, match="changed during"):
        operation()
    assert not args.results.exists()


@pytest.mark.usefixtures("modules")
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX search runner uses resource")
def test_search_rechecks_encoder_when_search_backend_is_unchanged(monkeypatch):
    search = importlib.import_module("bench_search_compare")
    monkeypatch.setattr(search, "runtime_identity", lambda: "changed-encoder")
    with pytest.raises(ValueError, match="encoder runtime changed"):
        search.verify_encoder_runtime({"encoder_runtime_fingerprint": "initial-encoder"})
