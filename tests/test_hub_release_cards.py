"""Option B changes only release README bytes; vector files keep their exact SHA256."""

import importlib
import json
import subprocess
from pathlib import Path

import pytest

from support.hub_classics import delivery_fixture

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(not BENCHMARKS.exists(), reason="benchmarks absent")


@pytest.fixture
def release(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("hub_release_cards")


@pytest.fixture
def basis(release, monkeypatch, tmp_path):
    cards = importlib.import_module("hub_classic_cards")
    publication = importlib.import_module("hub_publication")
    sources = importlib.import_module("hub_classic_sources")
    monkeypatch.setattr(publication, "local_probe", lambda *_: {"synthetic_graph_fixture": True})
    entries = []
    for name in sources.SOURCES:
        parent = tmp_path / name
        parent.mkdir()
        models, records, verification, fit, _ = delivery_fixture(monkeypatch, parent, name)
        for record in records:
            folder = models / record["repo"].split("/")[1]
            current_fit = fit if record["method"] == "sif10" else None
            (folder / "README.md").write_text(
                cards.render(folder, record, verification, current_fit)
            )
            (folder / ".gitattributes").write_bytes(sources.ATTRIBUTES)
            if current_fit is not None:
                (folder / "preparation.json").write_text(json.dumps(current_fit, indent=2) + "\n")
            entries.append(publication.publication_entry(models, record, current_fit, []))
    manifest = {
        "repositories": entries,
        "total_bytes": sum(e["total_bytes"] for e in entries),
        "approved": False,
        "hub_writes_performed": False,
    }
    path = tmp_path / "basis.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    monkeypatch.setattr(
        release,
        "release_identity",
        lambda: {"version": "2.0.0", "tag": "v2.0.0", "tag_commit": "a" * 40},
    )
    monkeypatch.setattr(release, "check_runtime", lambda *_: None)
    return path, manifest


def test_all_30_release_views_change_only_readme_and_keep_models(release, basis, tmp_path):
    path, before = basis
    output = tmp_path / "release-2.0.0"
    after, receipt = release.prepare_release(path, output, release.file_sha256(path))
    assert len(after["repositories"]) == 30
    for old, new in zip(before["repositories"], after["repositories"], strict=True):
        assert new["publication_hold"] is None
        assert new["metadata"] == old["metadata"]
        assert new["probe"] == old["probe"]
        assert new["python_example"] == old["python_example"]
        old_folder, new_folder = Path(old["local_folder"]), Path(new["local_folder"])
        changed = {name for name in old["files"] if old["files"][name] != new["files"][name]}
        assert changed == {"README.md"}
        for name, spec in old["files"].items():
            if name != "README.md":
                assert (old_folder / name).stat().st_ino == (new_folder / name).stat().st_ino
                assert release.file_sha256(new_folder / name) == spec["sha256"]
        assert release.file_sha256(old_folder / "README.md") == old["files"]["README.md"]["sha256"]
        text = (new_folder / "README.md").read_text()
        assert "pip install fse\n" in text
        assert "requires fse 2.0.0 or later" in text
        assert "currently in development" not in text
        assert "git+https" not in text
        assert (output / "cards" / f"{new_folder.name}.md").read_text() == text
    assert receipt["publication_manifest_sha256"] == release.file_sha256(
        output / "publication-manifest.json"
    )
    assert all(row["changed_payload_files"] == ["README.md"] for row in receipt["repositories"])
    assert after["release"]["basis_manifest_sha256"] == release.file_sha256(path)
    assert after["approved"] is False
    assert after["hub_writes_performed"] is False


def test_changed_non_card_payload_fails_before_any_output(release, basis, tmp_path):
    path, manifest = basis
    folder = Path(manifest["repositories"][0]["local_folder"])
    (folder / "config.json").write_text("changed")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="differs from its basis"):
        release.prepare_release(path, output, release.file_sha256(path))
    assert not output.exists()


def test_release_view_never_reuses_an_existing_output(release, basis, tmp_path):
    path, _ = basis
    with pytest.raises(FileExistsError, match="never overwritten"):
        release.prepare_release(path, tmp_path, release.file_sha256(path))


def test_release_identity_rejects_development_runtime(release, monkeypatch):
    monkeypatch.setattr(release, "version", lambda _: "2.0.0.dev1")
    with pytest.raises(ValueError, match=r"installed fse 2\.0\.0 release"):
        release.release_identity()


def test_release_identity_requires_remote_tag(release, monkeypatch):
    monkeypatch.setattr(release, "version", lambda _: "2.0.0")
    monkeypatch.setattr(release.fse, "__version__", "2.0.0")
    monkeypatch.setattr(release.subprocess, "check_output", lambda *_a, **_kw: "")
    with pytest.raises(ValueError, match=r"no valid v2\.0\.0 release tag"):
        release.release_identity()


def test_release_identity_resolves_annotated_tag_commit(release, monkeypatch):
    monkeypatch.setattr(release, "version", lambda _: "2.0.0")
    monkeypatch.setattr(release.fse, "__version__", "2.0.0")
    monkeypatch.setattr(
        release.subprocess,
        "check_output",
        lambda *_a, **_kw: "b" * 40 + "\trefs/tags/v2.0.0\n" + "a" * 40 + "\trefs/tags/v2.0.0^{}\n",
    )
    assert release.release_identity() == {
        "version": "2.0.0",
        "tag": "v2.0.0",
        "tag_commit": "a" * 40,
    }


def test_remote_tag_command_failure_stops_before_rendering(release, monkeypatch):
    monkeypatch.setattr(release, "version", lambda _: "2.0.0")
    monkeypatch.setattr(release.fse, "__version__", "2.0.0")

    def unavailable(*_args, **_kwargs):
        raise subprocess.CalledProcessError(2, "git ls-remote")

    monkeypatch.setattr(release.subprocess, "check_output", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        release.release_identity()


def test_wrong_basis_hash_stops_before_release_or_output(release, basis, tmp_path):
    path, _ = basis
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="approved SHA256"):
        release.prepare_release(path, output, "0" * 64)
    assert not output.exists()


def test_release_runtime_failure_leaves_original_payload_and_no_output(
    release, basis, monkeypatch, tmp_path
):
    path, manifest = basis

    def incompatible(_entry):
        raise ValueError("release runtime does not reproduce the approved local probe")

    monkeypatch.setattr(release, "check_runtime", incompatible)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="does not reproduce"):
        release.prepare_release(path, output, release.file_sha256(path))
    assert not output.exists()
    entry = manifest["repositories"][0]
    model = Path(entry["local_folder"]) / "model.safetensors"
    assert release.file_sha256(model) == entry["files"]["model.safetensors"]["sha256"]
