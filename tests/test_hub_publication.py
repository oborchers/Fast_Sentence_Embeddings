"""PUBLIC payloads and fresh-cache evidence reject changed files or missing repos."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(not BENCHMARKS.exists(), reason="benchmarks absent")


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return SimpleNamespace(
        publication=importlib.import_module("hub_publication"),
        readback=importlib.import_module("hub_public_readback"),
        runner=importlib.import_module("verify_published_hub"),
    )


def test_only_literal_read_only_card_pattern_is_executed(modules):
    code = 'import fse\nencoder = fse.Encoder.from_pretrained("fse/example")\nvectors = encoder.encode(["A sentence."])'
    card = f"```python\n{code}\n```"
    assert modules.publication.card_example(card, "fse/example") == code
    for changed in (code + '\nprint("extra action")', code.replace("fse/example", "fse/other")):
        with pytest.raises(ValueError, match="read-only"):
            modules.publication.card_example(f"```python\n{changed}\n```", "fse/example")


def test_file_sha256_covers_exact_payload_bytes(modules, tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"abc")
    assert modules.publication.file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    path.write_bytes(b"abd")
    assert modules.publication.file_sha256(path) != modules.publication.sha256_bytes(b"abc")


def test_receipts_require_every_approved_repo_and_exact_manifest(modules, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    entry = {"repo": "fse/example"}
    manifest = {"repositories": [entry]}
    receipts = {
        "publication_manifest_sha256": modules.publication.file_sha256(path),
        "approved_repositories": [entry["repo"]],
        "repositories": [{"repo": entry["repo"], "revision": "a" * 40}],
    }
    assert modules.runner.uploaded_entries(manifest, receipts, path) == [(entry, "a" * 40)]
    receipts["approved_repositories"].append("fse/missing")
    with pytest.raises(ValueError, match="every approved repo"):
        modules.runner.uploaded_entries(manifest, receipts, path)
    receipts["publication_manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="exact approved manifest"):
        modules.runner.uploaded_entries(manifest, receipts, path)


def test_owned_cache_cleanup_refuses_unexpected_files(modules, tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    owner = {"repo": "fse/example", "path": str(case)}
    (case / "owner.json").write_text(json.dumps(owner))
    (case / "hf-home").mkdir()
    unexpected = case / "evidence.json"
    unexpected.write_text("preserve")
    with pytest.raises(ValueError, match="unexpected files"):
        modules.runner.remove_owned_cache(case, owner)
    assert unexpected.read_text() == "preserve"
    unexpected.unlink()
    assert modules.runner.remove_owned_cache(case, owner) > 0
    assert not case.exists()


def test_child_environment_uses_only_own_cache_and_no_token(modules, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-only")
    monkeypatch.setenv("HF_TOKEN_PATH", "/unused")
    env = modules.runner.cache_environment(tmp_path)
    assert "HF_TOKEN" not in env
    assert "HF_TOKEN_PATH" not in env
    assert env["HF_HOME"] == str(tmp_path)
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hub")
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_readback_records_each_check_and_keeps_failure_evidence(modules, monkeypatch, tmp_path):
    readback = modules.readback
    entry = {
        "repo": "fse/example",
        "local_folder": str(tmp_path),
        "method": "raw",
        "metadata": {"dimension": 3},
        "python_example_sha256": "test",
        "probe": {},
    }
    monkeypatch.setattr(readback, "check_public_info", lambda *_: None)
    monkeypatch.setattr(readback, "execute_example", lambda *_: object())
    monkeypatch.setattr(readback, "check_cache_revision", lambda *_: None)
    monkeypatch.setattr(readback, "download_and_check", lambda *_: ({}, {"checked": True}))
    monkeypatch.setattr(readback, "metadata_of", lambda *_: entry["metadata"])
    monkeypatch.setattr(readback, "run_probe", lambda *_: {"difference_points": 0.0})
    row = readback.check_uploaded_repo(entry, "a" * 40, tmp_path)
    assert row["status"] == "passed"
    assert all(row[flag] is True for flag in readback.FLAGS)
    monkeypatch.setattr(readback, "metadata_of", lambda *_: {"dimension": 4})
    row = readback.check_uploaded_repo(entry, "a" * 40, tmp_path)
    assert row["status"] == "failed"
    assert row["metadata_ok"] is False
    assert row["files_ok"] is True
    assert row["load_ok"] is True
    assert row["run_ok"] is None
    assert row["metadata"] == {"dimension": 4}
    assert row["hub_writes_performed"] is False


def test_nonempty_cache_fails_before_network(modules, tmp_path):
    (tmp_path / "not-empty").write_text("x")
    row = modules.readback.check_uploaded_repo({"repo": "fse/example"}, "a" * 40, tmp_path)
    assert row["error"]["stage"] == "fresh-cache"
    assert row["status"] == "failed"


@pytest.mark.parametrize("problem", ["shape", "dtype", "nan"])
def test_probe_rejects_invalid_vectors(modules, problem):
    rows = np.ones((512, 3), dtype=np.float32)
    if problem == "shape":
        rows = rows[:511]
    elif problem == "dtype":
        rows = rows.astype(np.float16)
    else:
        rows[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"shape|finite float32"):
        modules.publication.validate_vectors(rows, 512, 3)
