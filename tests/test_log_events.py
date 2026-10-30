"""The events of fse at their call sites (ARCHITECTURE.md, "Logging"): artifacts, the Hub
and the cleanup of temporary files."""

import errno
import json
import logging
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

import fse
import fse._hub as hub
from fse._container import _hash
from support import model2vec as mv
from support import sentence_transformers as st
from support.encoder import finished
from support.hub import FakeHub
from support.logs import captured, expect_log


def test_save_load_and_verify_of_an_encoder(tmp_path):
    encoder = finished()
    path = tmp_path / "e.safetensors"
    with expect_log("artifact.saved", kind="encoder", fingerprint=encoder.fingerprint) as saved:
        encoder.save(path)
    assert saved[0].fse_bytes == path.stat().st_size
    with expect_log("artifact.loaded", kind="encoder", mmap=True, bytes=path.stat().st_size):
        loaded = fse.Encoder.load(path, mmap=True)
    with expect_log("artifact.verified", kind="encoder", bytes=path.stat().st_size):
        loaded.verify()
    assert saved[0].name == "fse.artifact"
    assert saved[0].levelno == logging.INFO


def test_a_converted_model_and_its_ignored_keys(tmp_path):
    pytest.importorskip("model2vec")
    folder = mv.save_model(tmp_path / "m2v")
    config = json.loads((folder / "config.json").read_text())
    config["model_type"] = "model2vec"
    (folder / "config.json").write_text(json.dumps(config))
    with captured() as records:
        encoder = fse.Encoder.from_pretrained(folder)
    events = {r.fse_event: r for r in records}
    assert events["artifact.ignored"].fse_keys == ["model_type"]
    assert events["artifact.ignored"].fse_source == "model2vec"
    converted = events["artifact.converted"]
    assert (converted.fse_kind, converted.fse_source) == ("encoder", "model2vec")
    assert (converted.fse_rows, converted.fse_dim) == (
        len(encoder.word_vectors),
        encoder.output_dim,
    )
    assert converted.fse_tokenizer == encoder.tokenizer


def test_the_ignored_keys_of_a_sentence_transformers_model(tmp_path):
    with expect_log(
        "artifact.ignored", source="sentence_transformers", keys=["similarity_fn_name"]
    ):
        fse.Encoder.from_pretrained(st.write_model(tmp_path / "st"))


def test_word_vectors_of_gensim():
    gensim = pytest.importorskip("gensim")
    from fse.gensim import from_gensim

    kv = gensim.models.KeyedVectors(4)
    kv.add_vectors(["a", "b"], np.ones((2, 4), np.float32))
    with expect_log("artifact.converted", kind="word_vectors", source="gensim", rows=2, dim=4):
        from_gensim(kv, dtype="float16")


def test_a_head_of_scikit_learn():
    linear_model = pytest.importorskip("sklearn.linear_model")
    encoder = finished()
    x = encoder.encode([f"w{i} w{i + 1}" for i in range(20)])
    y = np.arange(20, dtype=np.float64)
    model = linear_model.LinearRegression().fit(x, y)
    with expect_log("artifact.converted", kind="linear_head", source="sklearn", outputs=1):
        fse.Regressor.from_sklearn(encoder, model)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="the cache needs links")
def test_a_repository_is_resolved_to_its_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakeHub("acme/shared", tmp_path / "cache")
    monkeypatch.setattr(hub, "_library", fake.library)
    finished().push_to_hub("acme/shared", license="mit")
    with captured() as records:
        fse.Encoder.from_pretrained("acme/shared", cache_dir=tmp_path / "cache")
    (resolved,) = [r for r in records if r.fse_event == "hub.resolved"]
    assert resolved.fse_commit == fake.branches["main"]
    assert resolved.fse_offline is False
    files = [r for r in records if r.fse_event == "hub.file"]
    assert {r.fse_commit for r in files} == {resolved.fse_commit}
    assert "lfs_sha256" in {r.fse_pin for r in files}


def _refusing(monkeypatch, target):
    """Make the removal of temporary files fail with EACCES."""

    def refuse(path, *_args, **_kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(target[0], target[1], refuse)


@pytest.mark.parametrize("dtype", ["safetensors", "npy"])
def test_a_failed_cleanup_never_replaces_the_error(tmp_path, monkeypatch, dtype):
    from fse._container import NpyContainer, SafetensorsContainer

    tensors = {"a": np.ones(3, np.float32)}
    container, path = (
        (SafetensorsContainer(), tmp_path / "x.safetensors")
        if dtype == "safetensors"
        else (NpyContainer(), tmp_path / "x")
    )

    def broken(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    if dtype == "safetensors":
        monkeypatch.setattr("fse._container._safetensors._write_data", broken)
        _refusing(monkeypatch, (Path, "unlink"))
    else:
        monkeypatch.setattr(NpyContainer, "_write_tensor", staticmethod(broken))
        _refusing(monkeypatch, (shutil, "rmtree"))
    with (
        expect_log("artifact.cleanup_failed", error="PermissionError", errno=errno.EACCES),
        pytest.raises(OSError, match="No space left") as raised,
    ):
        container.write(path, tensors, {"fse.kind": "test"})
    assert raised.value.errno == errno.ENOSPC
    (note,) = raised.value.__notes__
    assert note.startswith("fse could not remove the temporary ")


def test_a_temporary_file_already_gone_counts_as_removed(tmp_path):
    from fse._container import remove_temporary

    assert remove_temporary(tmp_path / "gone") is None
    folder = tmp_path / "dir"
    (folder / "sub").mkdir(parents=True)
    assert remove_temporary(folder) is None
    assert not folder.exists()


def test_records_cost_no_hashing_when_disabled(tmp_path, monkeypatch):
    """The fields of artifact records come from what a header states: logging at DEBUG
    hashes as often as logging off."""
    counts = []
    original = _hash.Hasher.update

    def counted(self, data):
        counts.append(1)
        return original(self, data)

    monkeypatch.setattr(_hash.Hasher, "update", counted)

    def run(name):
        counts.clear()
        encoder = finished(seed=3)
        encoder.save(tmp_path / f"{name}.safetensors")
        assert fse.Encoder.load(tmp_path / f"{name}.safetensors").fingerprint
        return len(counts)

    quiet = run("quiet")
    with captured():
        loud = run("loud")
    assert loud == quiet > 0


def test_the_plan_and_summary_of_a_run_in_the_calling_thread():
    encoder = finished()
    with captured() as records:
        encoder.encode([f"w{i}" for i in range(10)], workers=2)
    events = {r.fse_event: r for r in records}
    plan, done = events["runtime.plan"], events["runtime.done"]
    assert (plan.fse_source, plan.fse_rows, plan.fse_workers) == ("StringSource", 10, 2)
    assert (plan.fse_backend, plan.fse_dtype, plan.fse_sink) == ("cpu", "float32", None)
    assert (done.fse_rows, done.fse_tasks) == (10, 1)
    assert plan.threadName == done.threadName == threading.current_thread().name
    assert [r.fse_event for r in records].count("runtime.done") == 1


def test_the_default_workers_are_logged_when_they_change(monkeypatch):
    from fse import _cpus

    monkeypatch.setattr(_cpus, "_REPORTED", [])
    monkeypatch.setenv("PYTHON_CPU_COUNT", "3")
    with expect_log("runtime.workers", workers=3, origin="override") as records:
        _cpus.usable_cpus()
        _cpus.usable_cpus()
        monkeypatch.delenv("PYTHON_CPU_COUNT")
        _cpus.usable_cpus()
    assert len(records) == 2
    assert records[1].fse_origin in ("affinity", "cpu_count", "cgroup")


def test_a_head_folded_into_the_table():
    encoder = finished(normalize=False)
    weight = np.ones((2, encoder.output_dim), np.float32)
    with expect_log("head.folded", outputs=2, rows=len(encoder.word_vectors), backend="cpu"):
        head = fse.Regressor.from_arrays(encoder, weight, np.zeros(2, np.float32), normalized=False)
        head.predict(["w1 w2"])
