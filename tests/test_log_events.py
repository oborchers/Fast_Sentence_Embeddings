"""The events of fse at their call sites (docs/architecture/logging.md, "Logging"): artifacts, the Hub
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


TEXTS = [f"w{i} w{i + 1}" for i in range(40)]


def test_the_records_of_a_sink_and_a_store(tmp_path):
    encoder = finished()
    with captured() as records:
        with fse.ShardSink(tmp_path / "s", encoder, shard_rows=16) as sink:
            encoder.encode(TEXTS, sink=sink, workers=1)
        store = fse.VectorStore.open(tmp_path / "s")
        store.verify()
        store.search(encoder.encode(TEXTS[:2]), k=3)
    events = {}
    for r in records:
        events.setdefault(r.fse_event, []).append(r)
    (opened,) = events["store.sink_opened"]
    assert (opened.fse_mode, opened.fse_rows, opened.fse_shard_rows) == ("new", 0, 16)
    published = events["store.shard_published"]
    assert [r.fse_rows for r in published] == [16, 16, 8]
    assert {r.threadName for r in published} == {"fse-shard-publish"}
    (closed,) = events["store.sink_closed"]
    assert (closed.fse_outcome, closed.fse_durable_rows, closed.fse_shards) == ("ok", 40, 3)
    assert closed.fse_dropped_rows == 0
    assert (events["store.opened"][0].fse_rows, events["store.opened"][0].fse_shards) == (40, 3)
    assert events["store.verified"][0].fse_rows == 40
    (search,) = events["store.search"]
    assert (search.fse_queries, search.fse_k, search.fse_mode) == (2, 3, "exact")


def _empty_shard(tmp_path, monkeypatch):
    """A sink with an empty shard not published, whose removal fails with EACCES."""
    encoder = finished()
    sink = fse.ShardSink(tmp_path / "s", encoder, shard_rows=16)
    encoder.encode(TEXTS, sink=sink, workers=1)
    sink._writable()  # a new shard that takes no row: dropped at close

    def refuse(path):
        return PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr("fse._store._shard.remove_temporary", refuse)
    return sink


def _shard_threads(sink):
    """The threads of ``sink`` still alive after they had 10 s to end (others of the process
    are not its). ``stop`` returns when a thread has said it ended, a moment before the
    thread itself is gone, which a loaded machine makes visible."""
    threads = [t._thread for t in (sink._hash, sink._io, sink._publisher)]
    for thread in threads:
        thread.join(10)
    return [t.name for t in threads if t.is_alive()]


def test_a_failed_cleanup_without_an_error_raises_after_the_threads(tmp_path, monkeypatch):
    sink = _empty_shard(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="Permission denied"):
        sink.close()
    assert _shard_threads(sink) == []
    assert sink.durable_rows == 40
    assert len(fse.VectorStore.open(tmp_path / "s")) == 40


def test_a_failed_cleanup_after_an_error_is_noted_and_logged(tmp_path, monkeypatch):
    sink = _empty_shard(tmp_path, monkeypatch)
    with (
        expect_log("store.cleanup_failed", error="PermissionError", errno=errno.EACCES),
        pytest.raises(ValueError, match="the error of the application") as raised,
        sink,
    ):
        raise ValueError("the error of the application")
    assert str(raised.value) == "the error of the application"
    (note,) = raised.value.__notes__
    assert note.startswith("fse could not remove the temporary ")
    assert _shard_threads(sink) == []


def test_a_header_names_no_other_kind_in_a_record(tmp_path):
    from fse._container import SafetensorsContainer

    path = tmp_path / "x.safetensors"
    with captured() as records:
        SafetensorsContainer().write(path, {"a": np.ones(2, np.float32)}, {"fse.kind": "/a/b"})
        SafetensorsContainer().open(path, mmap=False)
        SafetensorsContainer().verify(path)
    assert {r.fse_kind for r in records if r.fse_event.startswith("artifact.")} == {"other"}


def test_a_removal_that_cannot_be_checked_is_a_failure(tmp_path, monkeypatch):
    from fse._container import remove_temporary

    target = tmp_path / "t"
    target.mkdir()

    def refuse(path, *_args, **_kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(shutil, "rmtree", refuse)
    monkeypatch.setattr(os, "lstat", refuse)  # a folder above it that cannot be read
    assert isinstance(remove_temporary(target), PermissionError)


def test_a_failed_cleanup_after_a_reported_error_still_raises(tmp_path, monkeypatch):
    sink = _empty_shard(tmp_path, monkeypatch)

    def broken():
        raise OSError(errno.EIO, "the drive failed")

    sink._io.submit(broken)
    with pytest.raises(OSError, match="the drive failed"):
        sink._io.drain()  # the caller saw the error of the thread and goes on
    with pytest.raises(PermissionError, match="Permission denied"):
        sink.close()
    assert _shard_threads(sink) == []


@pytest.mark.usefixtures("python_sigint")
def test_a_failed_cleanup_is_noted_on_a_deferred_interrupt(tmp_path, monkeypatch):
    import signal

    sink = _empty_shard(tmp_path, monkeypatch)
    real_stop = sink._hash.stop

    def stop():
        signal.raise_signal(signal.SIGINT)  # Ctrl+C while close ends the threads
        real_stop()

    sink._hash.stop = stop
    with (
        expect_log("store.cleanup_failed", error="PermissionError"),
        pytest.raises(KeyboardInterrupt) as raised,
    ):
        sink.close()
    (note,) = raised.value.__notes__
    assert note.startswith("fse could not remove the temporary ")
    assert _shard_threads(sink) == []


def test_a_failed_publish_reports_its_cleanup_in_the_calling_thread(tmp_path, monkeypatch):
    from fse._store import _shard

    encoder = finished()

    def broken(*_args, **_kwargs):
        raise OSError(errno.EIO, "the drive failed")

    monkeypatch.setattr(_shard.ShardWriter, "finish", broken)
    sink = fse.ShardSink(tmp_path / "s", encoder, shard_rows=16)
    monkeypatch.setattr(
        "fse._store._shard.remove_temporary",
        lambda path: PermissionError(errno.EACCES, "Permission denied", str(path)),
    )
    with (
        expect_log("store.cleanup_failed", error="PermissionError") as records,
        pytest.raises(OSError, match="the drive failed") as raised,
    ):
        _encode_into(sink, encoder)
    assert {r.threadName for r in records} == {threading.current_thread().name}
    assert raised.value.__notes__
    assert _shard_threads(sink) == []


def _encode_into(sink, encoder):
    with sink:
        encoder.encode(TEXTS, sink=sink, workers=1)


def test_expect_log_leaves_other_warnings_unexpected():
    from fse import _log
    from support.logs import unexpected

    with captured(logging.WARNING, mark=False) as other, expect_log("store.search"):
        _log.emit("store.search", "fse store: search")
        _log.emit("runtime.arrow_copied", "fse runtime: copied", bytes=1)
    found = [r for r in other if r.fse_event == "runtime.arrow_copied"]
    assert unexpected(found) == ["fse.runtime WARNING: fse runtime: copied"]
    for record in found:  # this test expects it after all
        record.fse_expected = True


def test_every_failed_cleanup_is_raised_or_noted(tmp_path, monkeypatch):
    sink = _empty_shard(tmp_path, monkeypatch)
    sink._writers[-1].seal()  # another empty shard after this one
    sink._writable()
    with (
        expect_log("store.cleanup_failed", error="PermissionError"),
        pytest.raises(PermissionError, match="Permission denied") as raised,
    ):
        sink.close()
    (note,) = raised.value.__notes__  # the second removal, noted on the first
    assert note.startswith("fse could not remove the temporary ")


@pytest.mark.usefixtures("python_sigint")
def test_a_deferred_interrupt_after_an_error_gets_the_note_too(tmp_path, monkeypatch):
    import signal

    sink = _empty_shard(tmp_path, monkeypatch)
    real_stop = sink._hash.stop

    def stop():
        signal.raise_signal(signal.SIGINT)
        real_stop()

    sink._hash.stop = stop
    with (
        expect_log("store.cleanup_failed"),
        pytest.raises(KeyboardInterrupt) as raised,
    ):
        _fail_within(sink, ValueError("the error of the application"))
    (note,) = raised.value.__notes__
    assert note.startswith("fse could not remove the temporary ")
    assert raised.value.__context__.__notes__  # the error of the block has it too


def _fail_within(sink, error):
    with sink:
        raise error
