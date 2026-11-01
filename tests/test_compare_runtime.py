"""Each counterpart binds its own installed files, never the fse package identity."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("compare_runtime")


@pytest.fixture
def installed(runtime, tmp_path, monkeypatch):
    root = tmp_path / "model2vec"
    root.mkdir()
    for name in ("__init__.py", "native.so", "settings.json"):
        (root / name).write_bytes(b"installed counterpart content")
    metadata = tmp_path / "model2vec.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_text("Name: model2vec\nVersion: 0.10.0\n")
    files = [Path("model2vec") / name for name in ("__init__.py", "native.so", "settings.json")]
    package = SimpleNamespace(
        files=[*files, Path("model2vec.dist-info/METADATA")],
        version="0.10.0",
        locate_file=lambda p: tmp_path / p,
    )
    spec = SimpleNamespace(origin=str(root / "__init__.py"), submodule_search_locations=[str(root)])
    monkeypatch.setattr(runtime, "distribution", lambda _: package)
    monkeypatch.setattr(runtime, "find_spec", lambda _: spec)
    return root


def test_counterpart_uses_its_own_install_path_version_and_content(runtime, installed):
    record = runtime.backend_identity("model2vec")
    assert record["runtime_package"] == "model2vec"
    assert record["runtime_package_version"] == "0.10.0"
    assert record["package_location"] == str(installed / "__init__.py")
    assert record["runtime_fingerprint"] != runtime.runtime_identity()


@pytest.mark.parametrize("name", ["__init__.py", "native.so", "settings.json"])
def test_changed_installed_file_changes_backend_content_hash(runtime, installed, name):
    before = runtime.backend_identity("model2vec")
    (installed / name).write_bytes(b"changed counterpart content")
    after = runtime.backend_identity("model2vec")
    assert before["runtime_fingerprint"] != after["runtime_fingerprint"]
    with pytest.raises(ValueError, match="changed during"):
        runtime.verify_backend_identity("model2vec", before)


def test_counterpart_record_cannot_carry_fse_identity(runtime):
    with pytest.raises(ValueError, match="cannot carry another backend"):
        runtime.verify_backend_identity("model2vec", {"runtime_package": "fse"})


def test_rust_identity_is_the_verified_binary_not_python_fse(runtime, monkeypatch):
    build = importlib.import_module("compare_rust_build")
    monkeypatch.setattr(
        build,
        "verified_binary",
        lambda: (Path("/verified/rust-worker"), {"binary_digest": "rust-binary-content"}),
    )
    record = runtime.backend_identity("rust")
    assert record["runtime_package"] == "model2vec-rs"
    assert record["package_location"] == str(Path("/verified/rust-worker").resolve())
    assert record["runtime_fingerprint"] == "rust-binary-content"
    assert record["rust_worker_build"]["binary_digest"] == "rust-binary-content"
