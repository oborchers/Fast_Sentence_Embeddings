"""Archive restoration verifies bytes before creating any destination file."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def archive(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    module = importlib.import_module("evidence_files")
    payload = b"original measured value: 2.8326386960565858\n"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "evidence_commit": "a" * 40,
                "files": [
                    {
                        "path": "benchmarks/baseline/results/study.json",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=payload),
    )
    return module, manifest, tmp_path / "restored.json", payload


def test_restore_preserves_bytes_and_refuses_overwrite(archive):
    module, manifest, output, payload = archive
    module.restore(manifest, "benchmarks/baseline/results/study.json", output)
    assert output.read_bytes() == payload
    with pytest.raises(FileExistsError):
        module.restore(manifest, "benchmarks/baseline/results/study.json", output)


def test_corrupt_evidence_never_creates_output(archive, monkeypatch):
    module, manifest, output, _ = archive
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=b"altered"),
    )
    with pytest.raises(ValueError, match="size or SHA256"):
        module.restore(manifest, "benchmarks/baseline/results/study.json", output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "unknown"])
def test_only_manifest_paths_are_read(archive, name):
    module, manifest, output, _ = archive
    with pytest.raises(ValueError, match=r"relative|manifest entry"):
        module.restore(manifest, name, output)
    assert not output.exists()
