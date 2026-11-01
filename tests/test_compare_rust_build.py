"""A Rust worker is usable only with its matching source and binary manifest."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("change", ["binary", "source"])
def test_changed_build_input_requires_rebuild(monkeypatch, tmp_path, change):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    module = importlib.import_module("compare_rust_build")
    for name in ("Cargo.toml", "Cargo.lock"):
        (tmp_path / name).write_text("build fixture")
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/main.rs"
    source.write_text("fn main() {}")
    binary = tmp_path / "worker"
    binary.write_bytes(b"binary fixture")
    manifest = tmp_path / "build-identity.json"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "BINARY", binary)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    manifest.write_text(
        json.dumps({"inputs": module.build_inputs(), "binary_digest": module.file_digest(binary)})
    )
    assert module.verified_binary()[0] == binary
    (binary if change == "binary" else source).write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from its recorded locked build"):
        module.verified_binary()
