"""Read the independent binary affine models of sklearn's multilabel OneVsRestClassifier.

No sklearn import or fitting. The caller supplies its checked binary estimator reader.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from fse._validation import refuse_lost_input


def read(model: Any, binary: Callable) -> tuple:
    """Return W, b, the independent link, ordered classes and provenance."""
    if getattr(model, "multilabel_", None) is not True:
        raise ValueError("OneVsRestClassifier import requires fitted multilabel targets")
    classes = getattr(model, "classes_", None)
    estimators = getattr(model, "estimators_", None)
    binarizer = getattr(model, "label_binarizer_", None)
    refuse_lost_input(classes, "classes_")
    refuse_lost_input(getattr(binarizer, "classes_", None), "label_binarizer_.classes_")
    if not isinstance(classes, np.ndarray) or classes.ndim != 1 or not len(classes):
        raise ValueError("OneVsRestClassifier.classes_ must be a nonempty label vector")
    if not isinstance(estimators, list | tuple) or len(estimators) != len(classes):
        raise ValueError("OneVsRestClassifier needs exactly one fitted estimator per label")
    if getattr(binarizer, "y_type_", None) != "multilabel-indicator" or not np.array_equal(
        getattr(binarizer, "classes_", None), classes
    ):
        raise ValueError("OneVsRestClassifier label_binarizer_ disagrees with classes_")
    dim = getattr(model, "n_features_in_", None)
    if isinstance(dim, bool) or not isinstance(dim, int | np.integer) or dim < 1:
        raise ValueError("OneVsRestClassifier needs a valid fitted n_features_in_")
    parts = [_binary(estimator, binary, int(dim)) for estimator in estimators]
    weight = np.concatenate([part[0] for part in parts], axis=0)
    bias = np.concatenate([part[1] for part in parts])
    return weight, bias, "multilabel", classes.tolist(), "OneVsRestClassifier"


def _binary(estimator: Any, binary: Callable, dim: int) -> tuple:
    cls = type(estimator)
    if cls.__module__ == "sklearn.multiclass" and cls.__name__ == "_ConstantPredictor":
        return _constant(estimator, dim)
    found = binary(estimator, "classification")
    if found.weight.shape != (1, dim) or found.bias.shape != (1,):
        raise ValueError("each OneVsRestClassifier estimator must have one affine binary output")
    if found.link != "sigmoid" or found.labels != [0, 1]:
        raise ValueError(
            "each OneVsRestClassifier estimator needs sigmoid probabilities and classes [0, 1]; "
            "decision-only and calibration wrappers are not supported by this importer"
        )
    return found.weight, found.bias


def _constant(estimator: Any, dim: int) -> tuple:
    value = getattr(estimator, "y_", None)
    refuse_lost_input(value, "y_")
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (1,)
        or value.dtype.kind not in "iu"
        or value[0] not in (0, 1)
    ):
        raise ValueError("a constant OneVsRestClassifier estimator must predict 0 or 1")
    # exp@1(-104) rounds to zero: finite logits give exactly the constant probability.
    return np.zeros((1, dim), np.float64), np.array([104.0 if value[0] else -104.0])
