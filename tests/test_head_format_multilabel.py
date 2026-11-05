"""The strict linear_head 2 schema does not reinterpret version 1 files."""

import numpy as np
import pytest

import fse
from fse._container import CHECKSUM_PREFIX, checksum
from fse._contract import UnsupportedArtifactError
from fse._head._format import from_artifact
from support.encoder import finished


def _artifact():
    encoder = finished(normalize=False)
    model = fse.Classifier.from_arrays(
        encoder,
        np.ones((2, encoder.output_dim)),
        labels=["a", "b"],
        link="multilabel",
        thresholds={"b": 0.3},
    )
    tensors, metadata, _ = model._artifact()
    metadata.update({CHECKSUM_PREFIX + name: checksum(value) for name, value in tensors.items()})
    return tensors, metadata


def test_schema_and_encoder_fingerprint():
    tensors, metadata = _artifact()
    assert metadata["fse.format_version"] == "2"
    assert metadata["fse.head"] == "linear@2"
    assert metadata["fse.head.link"] == "multilabel@1"
    content, enc_tensors, enc_metadata, fingerprint = from_artifact(tensors, metadata)
    np.testing.assert_array_equal(content.thresholds, np.array([0.5, 0.3], np.float32))
    assert content.flat is False
    assert "head.thresholds" not in enc_tensors
    assert CHECKSUM_PREFIX + "head.thresholds" not in enc_metadata
    assert fingerprint == metadata["fse.encoder_fingerprint"]


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("head.thresholds", None, "needs the tensor head.thresholds"),
        ("head.thresholds", np.zeros(2), "F32"),
        ("head.thresholds", np.zeros((2, 1), np.float32), "F32"),
        ("head.thresholds", np.zeros(1, np.float32), "2 F32 values"),
        ("head.thresholds", np.array([0, np.nan], np.float32), "not finite"),
        ("head.thresholds", np.array([-0.1, 0.5], np.float32), "in \\[0, 1\\]"),
        ("head.thresholds", np.array([0.5, 1.1], np.float32), "in \\[0, 1\\]"),
    ],
)
def test_invalid_threshold_tensors(name, value, match):
    tensors, metadata = _artifact()
    if value is None:
        del tensors[name]
    else:
        tensors[name] = value
    with pytest.raises(ValueError, match=match):
        from_artifact(tensors, metadata)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("fse.format_version", "1", ValueError),
        ("fse.head", "linear@1", ValueError),
        ("fse.head.link", "softmax@1", ValueError),
        ("fse.head.flat", "true", ValueError),
        ("fse.head.task", "regression", ValueError),
        ("fse.head.labels", '["a","b","c"]', ValueError),
        ("fse.head", "linear@3", UnsupportedArtifactError),
        ("fse.head.link", "multilabel@2", UnsupportedArtifactError),
        ("fse.format_version", "3", UnsupportedArtifactError),
    ],
)
def test_mismatched_or_unknown_schema(key, value, error):
    tensors, metadata = _artifact()
    metadata[key] = value
    with pytest.raises(error):
        from_artifact(tensors, metadata)


def test_version_one_cannot_smuggle_a_threshold_tensor():
    tensors, metadata = _artifact()
    metadata.update(
        {"fse.format_version": "1", "fse.head": "linear@1", "fse.head.link": "softmax@1"}
    )
    with pytest.raises(ValueError, match="only for multilabel"):
        from_artifact(tensors, metadata)
