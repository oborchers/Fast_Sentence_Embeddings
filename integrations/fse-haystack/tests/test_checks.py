"""Constructor validation and local path semantics."""

import importlib
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

MODULE = importlib.import_module("haystack_integrations.components.embedders.fse")
ADAPTER = MODULE.FseTextEmbedder
MODEL = "model"


@pytest.mark.parametrize("workers", [0, -1, True, np.True_, 1.5, "2"])
def test_invalid_workers_before_loading(workers):
    with pytest.raises(ValueError, match="workers must be a positive int, not"):
        ADAPTER(workers=workers)


def test_numpy_workers_normalized(encoder):
    adapter = ADAPTER(encoder=encoder, workers=np.int64(2))
    assert type(adapter.workers) is int
    assert adapter.workers == 2


@pytest.mark.parametrize(
    ("options", "exception"),
    [
        ({"encoder": object()}, TypeError),
        ({"dtype": 1}, TypeError),
        ({"dtype": "int8"}, ValueError),
        ({"load_kwargs": []}, TypeError),
        ({"load_kwargs": {1: True}}, TypeError),
    ],
)
def test_bad_configuration(options, exception):
    with pytest.raises(exception):
        ADAPTER(**options)


class TextPath:
    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return str(self.path)


class BytesPath:
    def __fspath__(self):
        return b"namespace/model"


@pytest.mark.parametrize("value", [b"namespace/model", BytesPath(), 123])
def test_nontext_model_rejected(value):
    with pytest.raises(TypeError, match="model"):
        ADAPTER(**{MODEL: value})


def _query(adapter):
    if hasattr(adapter, "warm_up"):
        adapter.warm_up()
    result = adapter.run("hello")
    return result["embedding"] if isinstance(result, dict) else result


@pytest.mark.parametrize("wrapper", [Path, TextPath])
def test_local_path_artifact_and_reconstruction(encoder, tmp_path, monkeypatch, wrapper):
    monkeypatch.chdir(tmp_path)
    path = Path("namespace/model")
    path.parent.mkdir()
    encoder.save(path)
    adapter = ADAPTER(**{MODEL: wrapper(path)})
    expected = encoder.encode(["hello"]).tolist()[0]
    assert _query(adapter) == expected
    if hasattr(adapter, "to_dict"):
        data = adapter.to_dict()
        stored = data.get("init_parameters", data)
        assert os.path.isabs(stored[MODEL])
        monkeypatch.chdir(tmp_path.parent)
        assert _query(ADAPTER.from_dict(data)) == expected
        if MODEL == "model_name":
            assert _query(pickle.loads(pickle.dumps(adapter))) == expected


def test_missing_hub_shaped_path_stays_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Socket access is disabled: a Hub fallback would fail with SocketBlockedError.
    with pytest.raises(FileNotFoundError):
        _query(ADAPTER(**{MODEL: Path("namespace/missing")}))


def test_exports_only_adapters():
    expected = {"FseTextEmbedder", "FseDocumentEmbedder"}
    assert set(MODULE.__all__) == expected
    namespace = {}
    exec("from haystack_integrations.components.embedders.fse import *", namespace)
    assert set(namespace) - {"__builtins__"} == expected
