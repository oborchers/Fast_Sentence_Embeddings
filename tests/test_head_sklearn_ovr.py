"""Read OneVsRest's fitted affine binary heads without importing sklearn."""

from types import SimpleNamespace

import numpy as np
import pytest

from fse._head import _sklearn
from fse._head._sklearn_ovr import read


def _estimator(name, **fields):
    cls = type(
        name,
        (),
        {
            "__module__": "sklearn.multiclass"
            if name in ("OneVsRestClassifier", "_ConstantPredictor")
            else "sklearn.linear_model"
        },
    )
    model = cls()
    model.__dict__.update(fields)
    return model


def _model():
    classes = np.array([0, 1, 2])
    estimators = [
        _estimator(
            "SGDClassifier",
            coef_=np.ones((1, 4)),
            intercept_=np.zeros(1),
            classes_=np.array([0, 1]),
            loss="log_loss",
        )
        for _ in classes
    ]
    return _estimator(
        "OneVsRestClassifier",
        classes_=classes,
        estimators_=estimators,
        n_features_in_=4,
        multilabel_=True,
        label_binarizer_=SimpleNamespace(classes_=classes, y_type_="multilabel-indicator"),
    )


def test_reads_class_order_and_preserves_every_binary_head():
    model = _model()
    model.estimators_[1].coef_ *= 2
    got = _sklearn.read(model, "classification")
    assert got.link == "multilabel"
    assert got.labels == [0, 1, 2]
    np.testing.assert_array_equal(got.weight, [[1] * 4, [2] * 4, [1] * 4])


def test_constants_have_exact_zero_or_one_probabilities():
    from fse import _reference_head as ref
    from fse._contract import Link

    model = _model()
    model.estimators_[:2] = [
        _estimator("_ConstantPredictor", y_=np.array([value])) for value in (0, 1)
    ]
    found = _sklearn.read(model, "classification")
    assert not found.weight[:2].any()
    result = ref.proba(found.bias[None, :].astype(np.float32), Link.MULTILABEL)
    assert result[0, :2].tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("multilabel_", False, "multilabel targets"),
        ("estimators_", [], "exactly one"),
        ("classes_", np.zeros((1, 3)), "label vector"),
        ("n_features_in_", True, "n_features"),
        ("label_binarizer_", None, "disagrees"),
    ],
)
def test_invalid_wrapper_is_rejected(field, value, match):
    model = _model()
    setattr(model, field, value)
    with pytest.raises(ValueError, match=match):
        read(model, _sklearn._read)


@pytest.mark.parametrize("change", ["shape", "classes", "no probabilities", "constant"])
def test_invalid_binary_models_are_rejected(change):
    model = _model()
    item = model.estimators_[0]
    if change == "shape":
        item.coef_ = np.ones((1, 3))
    elif change == "classes":
        item.classes_ = np.array([1, 2])
    elif change == "constant":
        model.estimators_[0] = _estimator("_ConstantPredictor", y_=np.array([2]))
    else:
        item.loss = "hinge"
    with pytest.raises(
        ValueError, match=r"affine binary output|sigmoid probabilities|predict 0 or 1"
    ):
        read(model, _sklearn._read)


@pytest.mark.parametrize("normalize", [False, True])
def test_real_one_vs_rest_round_trip_and_named_cutoffs(normalize, tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import fse
    from support import head as support
    from support.encoder import finished

    encoder = finished(normalize=normalize)
    texts = [f"w{i} w{i + 1}" for i in range(60)]
    x = encoder.encode(texts)
    y = np.column_stack([np.arange(60) % 2, np.arange(60) % 3 == 0, np.arange(60) % 5 == 0]).astype(
        int
    )
    estimator = make_pipeline(
        StandardScaler(), OneVsRestClassifier(LogisticRegression(max_iter=500))
    )
    estimator.fit(x, y)
    model = fse.Classifier.from_sklearn(encoder, estimator, thresholds={2: 0.4})
    z = support.logits(encoder._spec, model.coef_, model.intercept_, texts, normalized=normalize)
    from fse import _reference_head as ref

    np.testing.assert_array_equal(
        model.predict_proba(texts).view(np.uint32), ref.sigmoid(z).view(np.uint32)
    )
    np.testing.assert_allclose(
        model.predict_proba(texts), estimator.predict_proba(x), rtol=3e-5, atol=3e-6
    )
    np.testing.assert_array_equal(
        model.predict(texts), ref.sigmoid(z) > np.array([0.5, 0.5, 0.4], np.float32)
    )
    # Compare source labels only away from the rounding-sized decision boundary.
    source = estimator.predict_proba(x)
    certain = np.abs(source - np.array([0.5, 0.5, 0.4])) > 1e-5
    expected = source > np.array([0.5, 0.5, 0.4])
    np.testing.assert_array_equal(model.predict(texts)[certain], expected[certain])
    path = tmp_path / "imported.safetensors"
    model.save(path)
    loaded = fse.Classifier.load(path)
    assert loaded.fingerprint == model.fingerprint
    np.testing.assert_array_equal(loaded.predict(texts), model.predict(texts))


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel"])
def test_explicit_import_names_keep_output_order(kind):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    import fse
    from support.encoder import finished

    encoder = finished(normalize=True)
    texts = [f"w{i}" for i in range(40)]
    x = encoder.encode(texts)
    if kind == "multilabel":
        y = np.column_stack([np.arange(40) % 2, np.arange(40) % 3 == 0]).astype(int)
        trained = OneVsRestClassifier(LogisticRegression(max_iter=500)).fit(x, y)
        names = ["sport", "news"]
    else:
        k = 2 if kind == "binary" else 3
        trained = LogisticRegression(max_iter=500).fit(x, np.arange(40) % k)
        names = [f"label_{i}" for i in range(k)]
    original = fse.Classifier.from_sklearn(encoder, trained)
    renamed = fse.Classifier.from_sklearn(encoder, trained, labels=names)
    assert renamed.classes_.tolist() == names
    for method in ("decision_function", "predict_proba"):
        np.testing.assert_array_equal(
            getattr(renamed, method)(texts).view(np.uint32),
            getattr(original, method)(texts).view(np.uint32),
        )
    assert renamed.fingerprint != original.fingerprint
    if kind == "multilabel":
        named = fse.Classifier.from_sklearn(
            encoder, trained, labels=names, thresholds={"sport": 0.4}
        )
        assert named.thresholds["sport"] == float(np.float32(0.4))
    for bad in (["missing"], ["same"] * len(names)):
        with pytest.raises(ValueError, match=r"needs|unique"):
            fse.Classifier.from_sklearn(encoder, trained, labels=bad)
