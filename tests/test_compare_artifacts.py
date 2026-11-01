"""Changed encoder content cannot retain a fitted head's preparation identity."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def artifacts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("compare_artifacts")


def test_head_requires_its_float32_training_encoder_or_bound_descendant(artifacts):
    training = {"source_model": "model", "source_revision": "pin", "encoder_fingerprint": "parent"}
    manifest = {"head_encoders": {"head": training}}
    quantized = {**training, "encoder_fingerprint": "int8", "float32_parent_fingerprint": "parent"}
    artifacts.check_head_encoder(manifest, "head", quantized)
    quantized["float32_parent_fingerprint"] = "another-parent"
    with pytest.raises(ValueError, match="not a descendant"):
        artifacts.check_head_encoder(manifest, "head", quantized)


def test_native_weight_change_invalidates_head_lineage(artifacts, tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"training weights")
    before = artifacts.native_encoder_identity("minishlab/potion-base-8M", tmp_path)
    path.write_bytes(b"different weights")
    after = artifacts.native_encoder_identity("minishlab/potion-base-8M", tmp_path)
    with pytest.raises(ValueError, match="not a descendant"):
        artifacts.check_head_encoder({"head_encoders": {"head": before}}, "head", after)


def test_bound_source_must_match_requested_model(artifacts, monkeypatch, tmp_path):
    bound = importlib.import_module("quality.artifact_manifest")
    monkeypatch.setattr(
        bound,
        "verified_source",
        lambda *_: {
            "source_model": "other/model",
            "source_revision": "pin",
            "table_dtype": "float32",
        },
    )
    with pytest.raises(ValueError, match="requested pinned model"):
        artifacts.checked_encoder(
            SimpleNamespace(fingerprint="fp"), tmp_path, "minishlab/potion-base-8M", "float32"
        )


def test_runtime_fingerprint_covers_native_binary(artifacts, monkeypatch, tmp_path):
    import fse

    package = tmp_path / "__init__.py"
    package.write_text("# package fixture\n")
    binary = tmp_path / "_native.so"
    binary.write_bytes(b"first extension")
    monkeypatch.setattr(fse, "__file__", str(package))
    before = artifacts.runtime_identity()
    binary.write_bytes(b"rebuilt extension")
    assert artifacts.runtime_identity() != before


def test_head_source_is_recorded_in_the_sealed_manifest(artifacts, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset_revision": "pinned"}))
    artifacts.record_head_encoder(tmp_path, "head", {"encoder_fingerprint": "training"})
    artifacts.seal_preparation(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["head_encoders"]["head"]["encoder_fingerprint"] == "training"
