"""Regression checks for the standalone recipe helpers, without recipe extras."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def ann():
    return _helper("_ann")


def _helper(name):
    path = Path(__file__).resolve().parents[1] / "recipes" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("argument", ["keys", "counts"])
@pytest.mark.parametrize("masked", [True, False])
def test_unsigned_candidates_refuse_masked_arrays(ann, argument, masked):
    values = {"keys": np.array([[0, 1]], dtype="u8"), "counts": np.array([2])}
    values[argument] = np.ma.array(values[argument], mask=masked)
    with pytest.raises(TypeError, match="unmasked integer"):
        ann.candidates_from_unsigned(**values, n_rows=3)


@pytest.mark.parametrize("argument", ["keys", "valid"])
@pytest.mark.parametrize("masked", [True, False])
def test_view_candidates_refuse_masked_arrays(ann, tmp_path, argument, masked):
    view = SimpleNamespace(paths=[tmp_path], members=[{"rows": 3, "first_row": 2}])
    values = {"keys": np.array([[0]], dtype="u8"), "valid": np.array([[True]])}
    values[argument] = np.ma.array(values[argument], mask=masked)
    with pytest.raises(TypeError, match="unmasked"):
        ann.view_candidates(view, tmp_path, **values)


@pytest.mark.parametrize("counts", [True, np.bool_(True), [True, 1], {1}, {"count": 1}])
def test_unsigned_candidates_reject_noninteger_counts_before_conversion(ann, counts):
    with pytest.raises(TypeError, match="unmasked integer"):
        ann.candidates_from_unsigned(np.array([[0]], dtype="u8"), counts, 3)


def test_candidates_keep_padding_and_view_offsets(ann, tmp_path):
    keys = np.array([[0, 1, 2**64 - 1]], dtype="u8")
    np.testing.assert_array_equal(ann.candidates_from_unsigned(keys, 2, 3), [[0, 1, -1]])
    view = SimpleNamespace(paths=[tmp_path], members=[{"rows": 3, "first_row": 2}])
    np.testing.assert_array_equal(
        ann.view_candidates(view, tmp_path, keys, np.array([[True, True, False]])),
        [[2, 3, -1]],
    )


def test_snapshot_does_not_return_metadata_only_shared_cache(monkeypatch, tmp_path):
    # The Hub can return a config-only snapshot for local_files_only=True.
    # Simulate that behavior, and an online download which supplies the missing weights.
    rerank = _helper("_rerank")
    partial, complete = tmp_path / "partial", tmp_path / "work" / "hub" / "complete"
    partial.mkdir()
    (partial / "config.json").write_text("{}")

    def download(repo, *, revision, local_files_only=False, cache_dir=None, token=None):
        assert (repo, revision) == ("acme/model", "a" * 40)
        if local_files_only:
            return str(partial)
        assert cache_dir == tmp_path / "work" / "hub"
        assert token is False
        complete.mkdir(parents=True)
        (complete / "config.json").write_text("{}")
        (complete / "model.safetensors").write_bytes(b"weights")
        return str(complete)

    monkeypatch.setattr(rerank, "snapshot_download", download)
    folder = Path(rerank.snapshot(SimpleNamespace(work=tmp_path / "work"), "acme/model", "a" * 40))
    assert (folder / "model.safetensors").is_file()
    assert list(partial.iterdir()) == [partial / "config.json"]


@pytest.mark.parametrize("git_state", ["checkout", "archive", "no git"])
def test_speed_provenance_supports_archives(monkeypatch, git_state):
    import subprocess
    from importlib.metadata import version

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "recipes"))
    speed = _helper("_speed")
    monkeypatch.setattr(speed.platform, "system", lambda: "Linux")
    monkeypatch.setattr(speed.platform, "platform", lambda: "Linux-test")

    def output(command, **_kwargs):
        if git_state == "no git":
            raise FileNotFoundError("git")
        if git_state == "archive":
            raise subprocess.CalledProcessError(128, command)
        return "a" * 40 + "\n"

    monkeypatch.setattr(speed.subprocess, "check_output", output)
    info = speed.provenance()
    assert info["fse_version"] == version("fse")
    assert info["fse_commit"] == ("a" * 40 if git_state == "checkout" else None)
