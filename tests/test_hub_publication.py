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
        evidence=importlib.import_module("hub_readback_evidence"),
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
    monkeypatch.setattr(
        readback,
        "run_probe",
        lambda *_: {"difference_points": 0.0, "tolerance_ok": True, "bitwise_ok": True},
    )
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


def test_literal_card_example_runs_public_loader_and_real_encoder(modules, monkeypatch, tmp_path):
    import fse

    encoder = fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "sentence", "."])
    calls = []

    def public_load(repo):
        calls.append(repo)
        return encoder

    monkeypatch.setattr(fse.Encoder, "from_pretrained", public_load)
    code = 'import fse\nencoder = fse.Encoder.from_pretrained("fse/example")\nvectors = encoder.encode(["A sentence."])'
    card = f"```python\n{code}\n```".encode()
    (tmp_path / "README.md").write_bytes(card)
    entry = {
        "repo": "fse/example",
        "python_example": code,
        "python_example_sha256": modules.publication.sha256_bytes(code.encode()),
        "files": {"README.md": {"sha256": modules.publication.sha256_bytes(card)}},
        "metadata": {"dimension": 3},
    }
    assert modules.readback.execute_example(entry, tmp_path) is encoder
    assert calls == ["fse/example"]
    (tmp_path / "README.md").write_bytes(card + b"changed")
    with pytest.raises(ValueError, match="card changed"):
        modules.readback.execute_example(entry, tmp_path)


def test_download_rejects_one_changed_payload_file(modules, monkeypatch, tmp_path):
    import huggingface_hub

    payload = tmp_path / "model.safetensors"
    payload.write_bytes(b"abc")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *_args, **_kw: payload)
    entry = {
        "repo": "fse/example",
        "files": {
            "model.safetensors": {"bytes": 3, "sha256": modules.publication.sha256_bytes(b"abc")}
        },
    }
    _, actual = modules.readback.download_and_check(entry, "a" * 40, tmp_path)
    assert actual == entry["files"]
    payload.write_bytes(b"abd")
    with pytest.raises(ValueError, match="differs from the approved build"):
        modules.readback.download_and_check(entry, "a" * 40, tmp_path)


def test_disk_guard_failure_becomes_durable_failed_row(modules, monkeypatch, tmp_path):
    def unavailable(*_args):
        raise RuntimeError("less than 25 GiB free")

    monkeypatch.setattr(modules.runner, "run_one", unavailable)
    args = SimpleNamespace(output=tmp_path)
    row = modules.runner.checked_run({"repo": "fse/example"}, "a" * 40, args)
    assert row["status"] == "failed"
    assert row["error"]["type"] == "RuntimeError"
    assert all(row[flag] is False for flag in modules.readback.FLAGS)


def test_failed_probe_keeps_observed_difference_and_vector_hash(modules, monkeypatch, tmp_path):
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
    monkeypatch.setattr(readback, "download_and_check", lambda *_: ({}, {}))
    monkeypatch.setattr(readback, "metadata_of", lambda *_: entry["metadata"])
    probe = {
        "difference_points": 0.5,
        "pearson_x100": 60.5,
        "vectors_xxh3": "observed",
        "tolerance_ok": False,
        "bitwise_ok": False,
    }
    monkeypatch.setattr(readback, "run_probe", lambda *_: probe)
    row = readback.check_uploaded_repo(entry, "a" * 40, tmp_path)
    assert row["status"] == "failed"
    assert row["run_ok"] is False
    assert row["difference_points"] == 0.5
    assert row["probe"] == probe
    assert "+0.50000000" in "\n".join(modules.runner.table([row]))


def test_failed_atomic_replacement_preserves_previous_receipt(modules, monkeypatch, tmp_path):
    evidence = modules.evidence
    path = tmp_path / "result.json"
    evidence.write_atomic(path, {"observed": 0.5})

    def interrupted(*_args):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(evidence.os, "replace", interrupted)
    with pytest.raises(OSError, match="injected replacement failure"):
        evidence.write_atomic(path, {"observed": 0.7})
    assert json.loads(path.read_text()) == {"observed": 0.5}
    assert list(tmp_path.iterdir()) == [path]


def test_timeout_preserves_published_child_measurements(modules, tmp_path):
    import subprocess

    result = tmp_path / "result.json"
    prior = {
        "repo": "fse/example",
        "status": "passed",
        "difference_points": 0.0001,
        "probe": {"pearson_x100": 60.0001},
        "files_ok": True,
    }
    modules.evidence.write_atomic(result, prior)
    row = modules.runner.failed_child(result, "fse/example", subprocess.TimeoutExpired([], 1))
    assert row["difference_points"] == prior["difference_points"]
    assert row["probe"] == prior["probe"]
    assert row["files_ok"] is True
    assert row["status"] == "failed"
    assert row["process_error"]["type"] == "TimeoutExpired"


def test_new_evidence_directories_sync_every_parent(modules, monkeypatch, tmp_path):
    synced = []
    monkeypatch.setattr(modules.evidence, "_sync_directory", synced.append)
    target = tmp_path / "new-parent" / "evidence"
    modules.evidence.mkdir_durable(target)
    assert target.is_dir()
    assert synced == [tmp_path, target.parent]


def test_output_created_after_precheck_cannot_be_reused(modules, monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    receipts = tmp_path / "receipts.json"
    manifest.write_text("{}")
    receipts.write_text("{}")
    output = tmp_path / "evidence"

    def concurrent_creation(*_args):
        output.mkdir()
        (output / "other-run.json").write_text("preserve")
        return []

    monkeypatch.setattr(modules.runner, "uploaded_entries", concurrent_creation)
    args = SimpleNamespace(
        output=output, scratch=tmp_path / "scratch", manifest=manifest, upload_receipts=receipts
    )
    with pytest.raises(FileExistsError):
        modules.runner.parent_run(args)
    assert (output / "other-run.json").read_text() == "preserve"
    assert {path.name for path in output.iterdir()} == {"other-run.json"}
