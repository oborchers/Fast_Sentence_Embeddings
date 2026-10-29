"""The inference helpers of linear models (fse._head._run)."""

import numpy as np
import pytest

import fse
from fse._head._run import RUNTIME, check_backend, check_output_sink, finish
from support.encoder import finished

F32 = np.float32


def _model(k=2):
    names = ["a", "b"] if k == 1 else [f"c{i}" for i in range(k)]
    return fse.Classifier.from_arrays(finished(normalize=False), np.ones((k, 6)), labels=names)


def test_every_output_has_a_runtime_output():
    assert RUNTIME == {
        "labels": "labels",
        "proba": "proba",
        "decision": "logits",
        "values": "logits",
    }


def test_check_backend():
    assert check_backend("numpy") == "numpy"
    with pytest.raises(ValueError, match="backend must be one of"):
        check_backend(3)


def test_finish_shapes_the_outputs():
    model = _model(1)
    assert finish(model, np.array([1, 0]), "labels").tolist() == ["b", "a"]
    assert finish(model, np.ones((3, 1), F32), "decision").shape == (3,)
    assert finish(model, np.ones((3, 2), F32), "proba").shape == (3, 2)
    assert finish(_model(3), np.ones((3, 3), F32), "decision").shape == (3, 3)


def test_check_output_sink(tmp_path):
    model = _model(3)
    check_output_sink(model, fse.MemorySink(3), "decision")
    with pytest.raises(ValueError, match="labels go to no sink"):
        check_output_sink(model, fse.MemorySink(1), "labels")
    with pytest.raises(ValueError, match="ShardSink stores the vectors"):
        check_output_sink(model, fse.ShardSink(tmp_path / "s", model.encoder), "proba")
    with pytest.raises(TypeError, match="target and commit"):
        check_output_sink(model, object(), "proba")
