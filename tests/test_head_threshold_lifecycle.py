"""Read and replace effective cutoffs without altering the original classifier."""

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse._container import CorruptArtifactError
from support.encoder import finished


def model():
    encoder = finished(normalize=False)
    return fse.Classifier.from_arrays(
        encoder,
        np.zeros((2, encoder.output_dim), np.float32),
        labels=["sport", "news"],
        link="multilabel",
        thresholds={"sport": 0.3, "news": 0.7},
    )


def test_snapshot_is_complete_and_readonly():
    original = model()
    snapshot = original.thresholds
    assert list(snapshot) == original.classes_.tolist()
    assert snapshot["sport"] == float(np.float32(0.3))
    with pytest.raises(TypeError):
        snapshot["sport"] = 0.9
    changed = original.with_thresholds({"sport": 0.6})
    assert changed is not original
    assert changed.thresholds["news"] == snapshot["news"]
    assert original.thresholds == snapshot
    assert changed.fingerprint != original.fingerprint
    assert original.with_thresholds({}).fingerprint == original.fingerprint
    np.testing.assert_array_equal(original.predict_proba([""]), changed.predict_proba([""]))
    assert original.predict([""]).tolist() == [[True, False]]
    assert changed.predict([""]).tolist() == [[False, False]]


@pytest.mark.parametrize("limits", [None, 0.5, [0.5, 0.5], (0.5, 0.5), np.array([0.5, 0.5])])
def test_full_replacement_and_explicit_reset(limits):
    assert dict(model().with_thresholds(limits).thresholds) == {"sport": 0.5, "news": 0.5}


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_changed_loaded_model_round_trip(tmp_path, standalone, dtype):
    original = model()
    if dtype != "float32":
        original = original.quantize(dtype)
    source = tmp_path / "original.safetensors"
    original.save(source, standalone=standalone)
    loaded = fse.Classifier.load(source, mmap=True)
    changed = loaded.with_thresholds({"sport": 0.9})
    target = tmp_path / "changed.safetensors"
    changed.save(target)
    restored = fse.Classifier.load(target)
    assert restored.thresholds == changed.thresholds
    assert restored.fingerprint == changed.fingerprint
    assert restored.encoder_fingerprint == loaded.encoder_fingerprint
    assert restored.standalone is standalone
    np.testing.assert_array_equal(restored.predict([""]), changed.predict([""]))
    assert loaded.thresholds == original.thresholds


@pytest.mark.parametrize("limits", [True, {"sport": "0.3"}, np.array([np.nan])])
def test_pydantic_checks_changes(limits):
    with pytest.raises(ValidationError):
        model().with_thresholds(limits)


def test_semantic_checks_and_nonmultilabel_errors():
    original = model()
    with pytest.raises(ValueError, match="unknown threshold label"):
        original.with_thresholds({"missing": 0.4})
    with pytest.raises(ValueError, match="2 values"):
        original.with_thresholds([0.3])
    single = fse.Classifier.from_arrays(original.encoder, original.coef_, labels=["a", "b"])
    with pytest.raises(ValueError, match="only for multilabel"):
        single.with_thresholds(None)
    with pytest.raises(ValueError, match="only for multilabel"):
        _ = single.thresholds


def test_changes_verify_loaded_tensors(tmp_path):
    from dataclasses import replace

    path = tmp_path / "original.safetensors"
    model().save(path)
    loaded = fse.Classifier.load(path)
    bias = loaded.intercept_
    bias[0] = 3
    loaded._content = replace(loaded._content, bias=bias)
    with pytest.raises(CorruptArtifactError, match=r"head.bias.*changed"):
        loaded.with_thresholds(None)
