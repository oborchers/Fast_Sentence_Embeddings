"""Caller-facing head types, checked without importing optional integrations."""

from typing import Any, assert_type

import numpy as np

import fse


def test_head_factory_types() -> None:
    encoder = fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])
    classifier = fse.Classifier.from_arrays(encoder, np.ones((2, 3), np.float32))
    regressor = fse.Regressor.from_arrays(encoder, np.ones(3, np.float32))
    assert_type(classifier, Any)
    assert_type(regressor, Any)
    assert isinstance(classifier, fse.Classifier)
    assert isinstance(regressor, fse.Regressor)
