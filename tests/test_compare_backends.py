"""Quantized-table benchmarks must load the table and keep output precision separate."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def backends(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("compare_backends")


def test_wrong_table_dtype_raises(backends, monkeypatch, tmp_path):
    import fse

    fake = SimpleNamespace(word_vectors=SimpleNamespace(dtype="float32"))
    monkeypatch.setattr(fse, "Encoder", SimpleNamespace(load=lambda _path: fake))
    with pytest.raises(ValueError, match="loaded table is float32, requested int8"):
        backends.fse_contender(1, "minishlab/potion-base-8M", "int8", tmp_path)


def test_int8_table_does_not_request_int8_outputs(backends, monkeypatch, tmp_path):
    import fse

    calls = []

    def encode(texts, **kwargs):
        calls.append(kwargs)
        return np.zeros((len(texts), 2), np.float32)

    artifact = tmp_path / "potion-base-8M-int8.safetensors"
    artifact.write_bytes(b"fixture")
    fake = SimpleNamespace(
        word_vectors=SimpleNamespace(dtype="int8"),
        fingerprint="test",
        encode=encode,
        tokenizer="fse_words@1",
    )
    monkeypatch.setattr(fse, "Encoder", SimpleNamespace(load=lambda _path: fake))
    monkeypatch.setattr(backends, "checked_encoder", lambda *_: {"source_model": "fixture"})
    run = backends.fse_contender(10, "minishlab/potion-base-8M", "int8", tmp_path)
    assert run(["a"]).dtype == np.float32
    assert run.table_dtype == "int8"
    assert run.file_bytes == len(b"fixture")
    assert calls == [{"workers": 10}]
