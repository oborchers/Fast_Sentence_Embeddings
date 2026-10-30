"""What fse's records hold and what they cannot change: no path, text, token or
repository id in any record at DEBUG, and the same results and errors whatever the
handlers of the application do (ARCHITECTURE.md, "Logging")."""

import errno
import logging
import os
from pathlib import Path

import numpy as np
import pytest

import fse
import fse._hub as hub
from fse import _log
from support import model2vec as mv
from support.encoder import finished
from support.hub import FakeHub
from support.logs import captured

# Inputs carry these; no record may contain one.
SECRETS = ("SECRET-PATH-7f3a", "SECRET-TEXT-91c2", "hf_SECRET-TOKEN", "SECRET-REPO", "secret-org")
REPO = "secret-org/SECRET-REPO"
TEXTS = [f"w{i} SECRET-TEXT-91c2 w{i + 1}" for i in range(40)]


def _values(record: logging.LogRecord) -> list[object]:
    """The message, args and fse fields of ``record``, nested containers flattened."""
    out: list[object] = []
    pending: list[object] = [record.getMessage(), record.msg, record.args]
    pending += [v for k, v in vars(record).items() if k.startswith("fse_")]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending += [*value.keys(), *value.values()]
        elif isinstance(value, list | tuple | set | frozenset):
            pending += list(value)
        else:
            out.append(value)
    return out


def _leaks(records):
    return [
        f"{r.fse_event}: {secret} in {value!r}"
        for r in records
        for value in _values(r)
        for secret in SECRETS
        if secret in str(value)
    ]


class NoArrow(list):
    def __arrow_c_stream__(self, requested_schema=None):
        raise ImportError("pyarrow is missing at SECRET-PATH-7f3a")


def _everything(root: Path, monkeypatch):
    """Most of what fse does, on inputs that carry the sentinels."""
    encoder = finished(normalize=False)
    encoder.save(root / "e.safetensors")
    loaded = fse.Encoder.load(root / "e.safetensors", mmap=True)
    loaded.verify()
    loaded.encode(TEXTS, workers=2)
    loaded.encode(NoArrow(TEXTS), workers=1)
    with fse.ShardSink(root / "store", loaded, shard_rows=16) as sink:
        loaded.encode(TEXTS, sink=sink, workers=1)
    store = fse.VectorStore.open(root / "store", encoder=loaded)
    store.verify()
    store.search(loaded.encode(TEXTS[:2]), k=3)
    head = fse.Regressor.from_arrays(
        loaded, np.ones((2, loaded.output_dim), np.float32), np.zeros(2), normalized=False
    )
    head.predict(TEXTS[:3])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakeHub(REPO, root / "cache")
    monkeypatch.setattr(hub, "_library", fake.library)
    encoder.push_to_hub(REPO, license="mit", token="hf_SECRET-TOKEN")
    fse.Encoder.from_pretrained(REPO, token="hf_SECRET-TOKEN", cache_dir=root / "cache")
    fse.Encoder.from_pretrained(REPO, cache_dir=root / "cache", local_files_only=True)
    if _has("model2vec"):
        fse.Encoder.from_pretrained(mv.save_model(root / "m2v"))


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="the Hub cache needs links")
def test_no_record_holds_a_path_text_token_or_repository(tmp_path, monkeypatch):
    root = tmp_path / "SECRET-PATH-7f3a"
    root.mkdir()
    with captured() as records:
        _everything(root, monkeypatch)
        _failed_cleanup(root, monkeypatch)
    seen = {r.fse_event for r in records}
    assert len(seen) >= 18, sorted(set(_log.EVENTS) - seen)
    assert _leaks(records) == []


def _failed_cleanup(root, monkeypatch):
    def refuse(path):
        return PermissionError(errno.EACCES, "Permission denied", str(path))

    def full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on SECRET-PATH-7f3a")

    monkeypatch.setattr("fse._container._common.remove_temporary", refuse)
    monkeypatch.setattr("fse._container._safetensors._write_data", full)
    with pytest.raises(OSError, match="No space left"):
        finished().save(root / "full.safetensors")


class Raising(logging.Handler):
    def emit(self, _record):
        raise RuntimeError("the handler of the application failed")


class Refusing(logging.Filter):
    def filter(self, _record):
        raise RuntimeError("the filter of the application failed")


@pytest.fixture(params=["handler", "filter", "factory"])
def broken_logging(request, monkeypatch):
    """Every logger of fse enabled and logging broken in one way."""
    monkeypatch.setattr(_log, "_reported", set())
    monkeypatch.setattr(logging, "raiseExceptions", False)  # nothing on stderr
    top = logging.getLogger("fse")
    with captured():
        if request.param == "handler":
            part = Raising()
            top.addHandler(part)
        elif request.param == "filter":
            part = Refusing()
            for name in _log.LOGGER_NAMES:
                logging.getLogger(name).addFilter(part)
        else:
            previous = logging.getLogRecordFactory()

            def factory(*_args, **_kwargs):
                raise RuntimeError("the factory of the application failed")

            logging.setLogRecordFactory(factory)
        try:
            yield
        finally:
            if request.param == "handler":
                top.removeHandler(part)
            elif request.param == "filter":
                for name in _log.LOGGER_NAMES:
                    logging.getLogger(name).removeFilter(part)
            else:
                logging.setLogRecordFactory(previous)


@pytest.mark.usefixtures("broken_logging")
def test_broken_logging_changes_no_result(tmp_path):
    encoder = finished()
    clean = finished()
    want = clean.encode(TEXTS, workers=2)
    assert encoder.encode(TEXTS, workers=2).tobytes() == want.tobytes()
    with fse.ShardSink(tmp_path / "s", encoder, shard_rows=16) as sink:
        encoder.encode(TEXTS, sink=sink, workers=1)
    assert sink.durable_rows == 40
    store = fse.VectorStore.open(tmp_path / "s")
    assert store.get_vectors().tobytes() == want.tobytes()


def _fail_within(sink, encoder, error):
    with sink:
        encoder.encode(TEXTS, sink=sink, workers=1)
        raise error


@pytest.mark.usefixtures("broken_logging")
def test_broken_logging_changes_no_error(tmp_path):
    encoder = finished()
    encoder.save(tmp_path / "e.safetensors")
    with pytest.raises(FileExistsError, match="never overwritten") as raised:
        encoder.save(tmp_path / "e.safetensors")
    assert raised.value.__cause__ is None
    error = ValueError("the error of the application")
    sink = fse.ShardSink(tmp_path / "s", encoder)
    with pytest.raises(ValueError, match="the error of the application") as raised:
        _fail_within(sink, encoder, error)
    assert raised.value is error
    assert not getattr(error, "__notes__", None)
    assert sink.durable_rows == 40
