"""Classifier labels retain their text and can be persisted."""

import numpy as np
import pytest

import fse


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("labels", [["a", "a\0"], ["\0", "\0\0"], ["", "a\0b"]])
def test_string_labels_survive_predictions_and_persistence(tmp_path, standalone, labels):
    encoder = fse.Encoder.from_arrays(
        np.array([[1], [-1]], dtype=np.float32), words=["pos", "neg"], normalize=False
    )
    model = fse.Classifier.from_arrays(encoder, np.array([[1]], dtype=np.float32), labels=labels)
    path = tmp_path / "head.safetensors"
    model.save(path, standalone=standalone)
    for classifier in (model, fse.Classifier.load(path)):
        assert classifier.classes_.tolist() == labels
        assert classifier.predict(["neg", "pos"]).tolist() == labels
        batches = classifier.predict_batches([["neg"], ["pos"]])
        assert [label for batch in batches for label in batch.outputs.tolist()] == labels


@pytest.mark.parametrize("label", ["\ud800", "a\udfff", "\ud800\udc00"])
def test_classifier_refuses_labels_without_utf8_at_construction(label):
    encoder = fse.Encoder.from_arrays(np.ones((1, 1), dtype=np.float32), words=["a"])
    with pytest.raises(ValueError, match=r"label.*UTF-8"):
        fse.Classifier.from_arrays(
            encoder, np.ones((1, 1), dtype=np.float32), labels=[label, "valid"]
        )
