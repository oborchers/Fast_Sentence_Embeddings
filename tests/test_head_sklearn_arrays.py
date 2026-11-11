"""scikit-learn source attributes retain their original types until validation."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fse import Classifier, Regressor
from support.encoder import finished


def _model(classifier=True):
    model = LogisticRegression() if classifier else LinearRegression()
    model.coef_ = np.ones((1, 6)) if classifier else np.ones(6)
    model.intercept_ = np.zeros(1) if classifier else 0.0
    model.n_features_in_ = 6
    if classifier:
        model.classes_ = np.array([0, 1])
    return model


@pytest.mark.parametrize("constructor", [Classifier, Regressor])
@pytest.mark.parametrize("field", ["coef_", "intercept_"])
@pytest.mark.parametrize(
    "invalid", ["mask-empty", "mask-active", "bool-array", "bool-scalar", "bool-object"]
)
def test_source_numbers_rejected_before_conversion(constructor, field, invalid):
    encoder = finished()
    model = _model(constructor is Classifier)
    value = np.asarray(getattr(model, field))
    bad = (
        np.ma.array(value, mask=invalid == "mask-active")
        if invalid.startswith("mask-")
        else (value.astype(bool).astype(object) if invalid == "bool-object" else value.astype(bool))
    )
    if invalid == "bool-scalar":
        bad = True
    setattr(model, field, bad)
    with pytest.raises(TypeError, match=rf"{field}.*(masked|bool|numeric)"):
        constructor.from_sklearn(encoder, model)


@pytest.mark.parametrize("mask", [False, True])
def test_source_classes_reject_masks(mask):
    model = _model()
    model.classes_ = np.ma.array(model.classes_, mask=mask)
    with pytest.raises(TypeError, match=r"classes_.*masked"):
        Classifier.from_sklearn(finished(), model)


@pytest.mark.parametrize("field", ["scale_", "mean_"])
@pytest.mark.parametrize(
    "invalid", ["mask-empty", "mask-active", "bool-array", "bool-scalar", "bool-object"]
)
def test_scaler_numbers_rejected_before_conversion(field, invalid):
    scaler = StandardScaler()
    scaler.scale_, scaler.mean_ = np.ones(6), np.zeros(6)
    scaler.n_features_in_ = 6
    value = getattr(scaler, field)
    setattr(
        scaler,
        field,
        np.ma.array(value, mask=invalid == "mask-active")
        if invalid.startswith("mask-")
        else (
            value.astype(bool).astype(object) if invalid == "bool-object" else value.astype(bool)
        ),
    )
    if invalid == "bool-scalar":
        setattr(scaler, field, True)
    model = Pipeline([("scaler", scaler), ("classifier", _model())])
    with pytest.raises(TypeError, match=rf"{field}.*(masked|bool|numeric)"):
        Classifier.from_sklearn(finished(), model)
