"""Platt-calibrated scikit-learn classifiers (fse._head._sklearn_calibrated): folded into the
head, compared with scikit-learn's own predictions within the bounds of
tests/support/sklearn_calibrated.py, and every other layout refused."""

import copy

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn import linear_model, pipeline, preprocessing, svm  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402

import fse  # noqa: E402
from fse import _reference_head as ref  # noqa: E402
from fse._contract import Head, Link  # noqa: E402
from fse._head import _sklearn as reader  # noqa: E402
from support import head as support  # noqa: E402
from support import head_bound  # noqa: E402
from support import sklearn_calibrated as cal  # noqa: E402
from support.encoder import finished  # noqa: E402
from support.runtime import texts  # noqa: E402

F32 = np.float32
SENTENCES = texts(700, seed=8)
VERSION = reader.installed_version()


def _data(encoder, classes=2, seed=0):
    x = encoder.encode(SENTENCES)
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((encoder.output_dim, classes))
    score = x.astype(np.float64) @ hidden + 0.8 * rng.standard_normal((len(x), classes))
    y = (score[:, 0] > 0).astype(int) if classes == 2 else np.argmax(score, axis=1)
    return x, y


def _calibrated(base, **kwargs):
    return CalibratedClassifierCV(base, method="sigmoid", ensemble=False, **kwargs)


BASES = [
    linear_model.LogisticRegression(),
    svm.LinearSVC(),
    linear_model.SGDClassifier(loss="hinge", random_state=0),
    linear_model.SGDClassifier(loss="modified_huber", random_state=0),
    linear_model.SGDClassifier(loss="log_loss", random_state=0),
    linear_model.RidgeClassifier(),
    linear_model.Perceptron(random_state=0),
]


def _ids(model):
    return type(model).__name__ + getattr(model, "loss", "")


def _check(clf, sk, encoder, x, scaler=None):
    """fse's logits, probabilities and labels of the calibrated ``sk`` against the exact
    head scikit-learn applies, within the bounds."""
    x64 = x.astype(np.float64)
    found = reader.read(sk, "classification")
    w64, b64 = found.weight, found.bias  # the float64 head that fse rounds to float32
    a, c = cal.calibrator(_calibrator_of(sk))
    inner = _estimator_of(sk)
    decision = inner.decision_function(x64).reshape(len(x), -1)
    z_star = -(a * decision + c)  # the ideal head, scikit-learn's own float64 evaluation
    spec = encoder._spec
    wf, bf = clf.coef_, clf.intercept_
    if clf.normalized:
        z = support.logits(spec, wf, bf, SENTENCES, normalized=True)
        bound = head_bound.unfolded(x, w64, b64, wf, bf, z)
    else:
        z = support.logits(spec, wf, bf, SENTENCES)
        table = support.folded_pool(spec.pool_input(), wf)
        batch = support.reference_batch(spec, SENTENCES)

        def rows(kind, i):
            return table[0][i] if kind == "word" else table[3][i]

        bound = head_bound.folded(spec, batch, rows, x, w64, b64, wf, bf, z)
    offset = None
    if scaler is not None and getattr(scaler, "with_mean", True):
        # The scaler folded inside: the bias loses its cancellation, |a| |W_f| |mean|.
        offset = np.abs(a)[None, :] * (np.abs(w64) @ np.abs(scaler.mean_))[None, :]
    bound = bound + cal.platt_distance(x64, w64, b64, c, offset)
    assert (np.abs(z - z_star) <= bound).all()
    return z, z_star, bound


def _calibrator_of(sk):
    return sk.steps[-1][1] if hasattr(sk, "steps") else sk


def _estimator_of(sk):
    """The fitted pipeline or classifier under the calibration that scikit-learn applies."""
    model = _calibrator_of(sk)
    inner = model.calibrated_classifiers_[0].estimator
    inner = (
        getattr(inner, "estimator", inner) if type(inner).__name__ == "FrozenEstimator" else inner
    )
    if hasattr(sk, "steps") and len(sk.steps) > 1:
        return pipeline.Pipeline([*sk.steps[:-1], ("last", inner)])
    return inner


def _compare(clf, sk, encoder, x, scaler=None):
    z, z_star, bound = _check(clf, sk, encoder, x, scaler)
    proba = clf.predict_proba(SENTENCES)
    want = sk.predict_proba(x.astype(np.float64))
    k = z.shape[1]
    labels = clf.predict(SENTENCES)
    if k == 1:
        assert clf.link == "sigmoid@1"
        tolerance = head_bound.proba_tolerance("sigmoid@1", bound, 1)
        assert (np.abs(proba - want).max(axis=1) <= tolerance).all()
    else:
        assert clf.link == "ovr@1"
        tolerance = cal.ovr_tolerance(z_star, bound, k)
        usable = ~np.isnan(tolerance)
        assert usable.any()
        assert (np.abs(proba - want).max(axis=1)[usable] <= tolerance[usable]).all()
    changed = labels != sk.predict(x.astype(np.float64))
    allowed = head_bound.allowed_label_changes(z_star, bound)
    assert not (changed & ~allowed).any()
    # predict is the first class of the largest probability, bit for bit.
    np.testing.assert_array_equal(labels, clf.classes_[np.argmax(proba, axis=1)])
    return z


@pytest.mark.parametrize("base", BASES, ids=_ids)
@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize("normalize", [False, True])
def test_calibrated_classifiers_match_scikit_learn(base, classes, normalize):
    encoder = finished(normalize=normalize)
    x, y = _data(encoder, classes)
    sk = cal.fit(_calibrated(base), x, y)
    clf = fse.Classifier.from_sklearn(encoder, sk)
    assert clf._head.rule is Head.LINEAR2
    assert clf.normalized == normalize
    assert clf.classes_.tolist() == sk.classes_.tolist()
    assert clf.n_outputs == (1 if classes == 2 else classes)
    _compare(clf, sk, encoder, x)


@pytest.mark.parametrize("where", ["inside", "outside"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("classes", [2, 3])
def test_a_standard_scaler_folds_inside_or_outside(where, dtype, classes):
    encoder = finished(normalize=False)
    x, y = _data(encoder, classes)
    x = x.astype(dtype) * dtype(3.0) + dtype(1.5)  # a scaler that does something
    encoder_x = encoder.encode(SENTENCES)
    scaler = preprocessing.StandardScaler()
    base = linear_model.LogisticRegression()
    if where == "outside":
        sk = pipeline.make_pipeline(scaler, _calibrated(base))
    else:
        sk = _calibrated(pipeline.make_pipeline(scaler, base))
    cal.fit(sk, x, y)
    clf = fse.Classifier.from_sklearn(encoder, sk)
    fitted = sk.steps[0][1] if where == "outside" else None
    if where == "inside":
        fitted = sk.calibrated_classifiers_[0].estimator.steps[0][1]
    # scikit-learn gets the vectors as fse gives them; the scaler of the test data is
    # applied by the head itself, so compare the head with the pipeline on fse's vectors.
    z, _z_star, _bound = _check(clf, sk, encoder, encoder_x, fitted)
    assert clf._head.rule is Head.LINEAR2
    proba = clf.predict_proba(SENTENCES)
    np.testing.assert_array_equal(clf.predict(SENTENCES), clf.classes_[np.argmax(proba, axis=1)])
    assert z.shape[0] == len(SENTENCES)


def test_the_head_is_the_float64_fold_rounded_once():
    encoder = finished(normalize=False)
    x, y = _data(encoder, 4)
    sk = cal.fit(_calibrated(svm.LinearSVC()), x, y)
    clf = fse.Classifier.from_sklearn(encoder, sk)
    a, c = cal.calibrator(sk)
    base = sk.calibrated_classifiers_[0].estimator
    weight = np.stack([-(a[j] * base.coef_[j]) for j in range(4)])
    bias = np.array([-(a[j] * base.intercept_[j] + c[j]) for j in range(4)])
    np.testing.assert_array_equal(clf.coef_, weight.astype(F32))
    np.testing.assert_array_equal(clf.intercept_, bias.astype(F32))


def test_a_positive_slope_flips_the_direction_and_a_zero_slope_is_constant():
    encoder = finished(normalize=False)
    x, y = _data(encoder, 2)
    sk = cal.fit(_calibrated(svm.LinearSVC()), x, y)
    calibrator = sk.calibrated_classifiers_[0].calibrators[0]
    for slope in (0.8, 0.0):
        calibrator.a_ = np.float64(slope)
        clf = fse.Classifier.from_sklearn(encoder, sk)
        probabilities = clf.predict_proba(SENTENCES)
        want = sk.predict_proba(x.astype(np.float64))
        assert np.abs(probabilities - want).max() < 1e-5
        if slope == 0.0:
            assert not clf.coef_.any()
            assert np.ptp(probabilities[:, 1]) == 0


def test_string_and_unsorted_integer_classes_keep_scikit_learns_order():
    encoder = finished(normalize=False)
    x, y = _data(encoder, 3)
    for names in (np.array(["pear", "apple", "fig"]), np.array([30, 5, 17])):
        sk = cal.fit(_calibrated(linear_model.LogisticRegression()), x, names[y])
        clf = fse.Classifier.from_sklearn(encoder, sk)
        assert clf.classes_.tolist() == sk.classes_.tolist()
        np.testing.assert_array_equal(
            clf.predict(SENTENCES), clf.classes_[clf.predict_proba(SENTENCES).argmax(axis=1)]
        )
    renamed = fse.Classifier.from_sklearn(encoder, sk, labels=["x", "y", "z"])
    assert renamed.classes_.tolist() == ["x", "y", "z"]


@pytest.mark.skipif(VERSION < (1, 6), reason="FrozenEstimator exists from scikit-learn 1.6")
@pytest.mark.parametrize("scaled", [False, True])
def test_a_frozen_estimator_is_unwrapped_once(scaled):
    from sklearn.frozen import FrozenEstimator

    encoder = finished(normalize=False)
    x, y = _data(encoder, 3)
    fitted = cal.fit(linear_model.LogisticRegression(), x, y)
    if scaled:
        fitted = cal.fit(pipeline.make_pipeline(preprocessing.StandardScaler(), fitted), x, y)
    sk = cal.fit(_calibrated(FrozenEstimator(fitted)), x, y)
    clf = fse.Classifier.from_sklearn(encoder, sk)
    proba = clf.predict_proba(SENTENCES)
    want = sk.predict_proba(x.astype(np.float64))
    assert np.abs(proba - want).max() < 1e-4
    assert clf.link == "ovr@1"


@pytest.mark.skipif(VERSION >= (1, 8), reason="cv='prefit' was removed in scikit-learn 1.8")
def test_a_prefit_estimator_with_ensemble_false_is_imported():
    encoder = finished(normalize=False)
    x, y = _data(encoder, 2)
    fitted = cal.fit(svm.LinearSVC(), x, y)
    sk = cal.fit(_calibrated(fitted, cv="prefit"), x, y)
    clf = fse.Classifier.from_sklearn(encoder, sk)
    assert (
        np.abs(clf.predict_proba(SENTENCES) - sk.predict_proba(x.astype(np.float64))).max() < 1e-5
    )


def _fitted(classes=3, **kwargs):
    encoder = finished(normalize=False)
    x, y = _data(encoder, classes)
    return encoder, cal.fit(_calibrated(linear_model.LogisticRegression(), **kwargs), x, y)


def test_saved_and_loaded_calibrated_heads_decide_alike(tmp_path):
    encoder, sk = _fitted()
    clf = fse.Classifier.from_sklearn(encoder, sk)
    clf.save(tmp_path / "m.safetensors")
    clf.save(tmp_path / "s.safetensors", standalone=True)
    clf.save_pretrained(tmp_path / "folder", license="mit")
    models = [
        fse.Classifier.load(tmp_path / "m.safetensors"),
        fse.Classifier.load(tmp_path / "m.safetensors", mmap=True),
        fse.Classifier.load(tmp_path / "s.safetensors"),
        fse.Classifier.from_pretrained(tmp_path / "folder"),
        clf.quantize("float16"),
    ]
    for model in models:
        assert model._head.rule is Head.LINEAR2
        assert model.link == "ovr@1"
    for model in models[:-1]:
        np.testing.assert_array_equal(model.predict(SENTENCES), clf.predict(SENTENCES))
        np.testing.assert_array_equal(
            model.predict_proba(SENTENCES).view(np.uint32),
            clf.predict_proba(SENTENCES).view(np.uint32),
        )
    card = (tmp_path / "folder" / "README.md").read_text()
    assert "Calibrated with Platt scaling (scikit-learn CalibratedClassifierCV" in card
    assert "estimator LogisticRegression" in card
    assert "- label rule: linear@2" in card


def test_the_logits_and_probabilities_are_the_reference_bits():
    encoder, sk = _fitted()
    clf = fse.Classifier.from_sklearn(encoder, sk)
    z = support.logits(encoder._spec, clf.coef_, clf.intercept_, SENTENCES)
    np.testing.assert_array_equal(
        clf.predict_proba(SENTENCES).view(np.uint32), ref.ovr(z).view(np.uint32)
    )
    np.testing.assert_array_equal(clf.predict(SENTENCES), clf.classes_[ref.predicted(z, Link.OVR)])


def test_source_and_the_binary_link_are_named():
    encoder, sk = _fitted(2)
    clf = fse.Classifier.from_sklearn(encoder, sk)
    assert clf._source == "CalibratedClassifierCV(LogisticRegression)"
    assert clf.link == "sigmoid@1"
    assert clf.predict_proba(SENTENCES).shape == (len(SENTENCES), 2)


# --- refusals ----------------------------------------------------------------------------


def _refused(model, match, encoder=None):
    with pytest.raises(ValueError, match=match):
        fse.Classifier.from_sklearn(encoder or finished(normalize=False), model)


def test_ensembles_and_other_methods_are_refused():
    encoder, sk = _fitted()
    ensemble = copy.deepcopy(sk)
    ensemble.ensemble = True
    _refused(ensemble, r"ensemble=True\) averages several sigmoid curves", encoder)
    auto = copy.deepcopy(sk)
    auto.ensemble = "auto"
    _refused(auto, r"ensemble='auto'\) is not imported; refit with ensemble=False", encoder)
    for method in ("isotonic", "temperature", "other", None):
        other = copy.deepcopy(sk)
        other.method = method
        _refused(other, rf"method={method!r}\) cannot be folded", encoder)


def test_a_model_that_is_not_fitted_is_refused():
    _refused(_calibrated(linear_model.LogisticRegression()), "is it fitted", None)


def test_wrong_layouts_are_refused_with_the_index():
    encoder, sk = _fitted()

    def mutated(change):
        model = copy.deepcopy(sk)
        change(model)
        return model

    _refused(
        mutated(lambda m: m.calibrated_classifiers_.append(m.calibrated_classifiers_[0])),
        "holds 2 calibrated classifiers; exactly one",
        encoder,
    )
    _refused(
        mutated(lambda m: setattr(m, "calibrated_classifiers_", [])),
        "holds 0 calibrated classifiers",
        encoder,
    )
    _refused(
        mutated(lambda m: setattr(m, "calibrated_classifiers_", ["x"])),
        "holds a str, an unknown layout",
        encoder,
    )
    _refused(
        mutated(lambda m: setattr(m.calibrated_classifiers_[0], "method", "isotonic")),
        "not a sigmoid calibration",
        encoder,
    )
    _refused(
        mutated(lambda m: m.calibrated_classifiers_[0].calibrators.pop()),
        "has 2 calibrators for 3 decision columns",
        encoder,
    )
    _refused(
        mutated(lambda m: m.calibrated_classifiers_[0].calibrators.__setitem__(1, object())),
        "calibrator 1 of CalibratedClassifierCV is a object, not a sigmoid calibration",
        encoder,
    )
    for value in (np.nan, np.inf, True, "1", None):
        _refused(
            mutated(
                lambda m, v=value: setattr(m.calibrated_classifiers_[0].calibrators[2], "a_", v)
            ),
            "calibrator 2 of CalibratedClassifierCV has a_ or b_ that is not a finite number",
            encoder,
        )
    _refused(
        mutated(lambda m: delattr(m.calibrated_classifiers_[0].calibrators[0], "b_")),
        "calibrator 0 of CalibratedClassifierCV has a_ or b_",
        encoder,
    )


def test_classes_that_differ_are_refused():
    encoder, sk = _fitted()
    for owner in ("model", "element", "estimator"):
        model = copy.deepcopy(sk)
        element = model.calibrated_classifiers_[0]
        target = {"model": model, "element": element, "estimator": element.estimator}[owner]
        attr = "classes" if owner == "element" else "classes_"
        setattr(target, attr, np.array([1, 0, 2]))
        _refused(
            model, "the classes of CalibratedClassifierCV and of its estimator differ", encoder
        )
    model = copy.deepcopy(sk)
    model.calibrated_classifiers_[0].estimator.classes_ = np.array([0.0, 1.0, 2.0])
    _refused(model, "the classes of CalibratedClassifierCV and of its estimator differ", encoder)


def test_only_linear_classifiers_may_be_calibrated():
    from sklearn.tree import DecisionTreeClassifier

    encoder = finished(normalize=False)
    x, y = _data(encoder, 3)
    tree = cal.fit(_calibrated(DecisionTreeClassifier(max_depth=2)), x, y)
    _refused(tree, "must be a linear classifier fse reads", encoder)
    nested = cal.fit(_calibrated(_calibrated(linear_model.LogisticRegression())), x, y)
    _refused(nested, "must be a linear classifier fse reads", encoder)
    wrapped = cal.fit(_calibrated(OneVsRestClassifier(linear_model.LogisticRegression())), x, y)
    _refused(wrapped, "must be a linear classifier fse reads", encoder)
    normalizer = pipeline.make_pipeline(
        preprocessing.Normalizer(), linear_model.LogisticRegression()
    )
    piped = cal.fit(_calibrated(normalizer), x, y)
    _refused(piped, "must be a linear classifier fse reads|Normalizer", encoder)


def test_thresholds_are_refused_for_a_calibrated_head():
    encoder, sk = _fitted()
    with pytest.raises(ValueError, match="thresholds are given only for multilabel"):
        fse.Classifier.from_sklearn(encoder, sk, thresholds=0.4)


def test_calibrated_models_in_a_one_vs_rest_classifier_are_refused():
    encoder = finished(normalize=False)
    x, _ = _data(encoder, 3)
    rng = np.random.default_rng(1)
    indicator = (rng.random((len(x), 3)) < 0.4).astype(int)
    indicator[:4] = np.eye(3, dtype=int)[[0, 1, 2, 0]]
    message = "OneVsRestClassifier estimators must not be calibrated"
    for make in (
        lambda: _calibrated(linear_model.LogisticRegression()),
        lambda: pipeline.make_pipeline(
            preprocessing.StandardScaler(), _calibrated(svm.LinearSVC())
        ),
    ):
        sk = cal.fit(OneVsRestClassifier(make()), x, indicator)
        _refused(sk, message, encoder)
    plain = cal.fit(OneVsRestClassifier(linear_model.LogisticRegression()), x, indicator)
    assert (
        fse.Classifier.from_sklearn(encoder, plain, labels=["a", "b", "c"]).link == "multilabel@1"
    )


@pytest.mark.skipif(VERSION < (1, 6), reason="FrozenEstimator exists from scikit-learn 1.6")
def test_a_frozen_estimator_is_read_only_inside_a_calibration():
    from sklearn.frozen import FrozenEstimator

    encoder = finished(normalize=False)
    x, y = _data(encoder, 3)
    frozen = FrozenEstimator(cal.fit(linear_model.LogisticRegression(), x, y))
    _refused(frozen, "FrozenEstimator|not a scikit-learn", encoder)
    _refused(pipeline.make_pipeline(frozen), "FrozenEstimator|not a scikit-learn", encoder)
    inner = cal.fit(_calibrated(frozen), x, y)
    twice = cal.fit(_calibrated(FrozenEstimator(inner)), x, y)
    _refused(twice, "must be a linear classifier fse reads", encoder)
    wrapped = cal.fit(_calibrated(FrozenEstimator(frozen)), x, y)
    _refused(wrapped, "must be a linear classifier fse reads", encoder)


def test_every_route_uses_the_probabilities_where_the_logits_disagree(tmp_path):
    """Saturated ovr: the logits 30 and 40 both give the probability 1.0. linear@2 takes the
    first class on texts, vectors and store routes, where linear@1 takes the larger logit."""
    encoder, sk = _fitted()
    sk = copy.deepcopy(sk)
    element = sk.calibrated_classifiers_[0]
    element.estimator.coef_ = np.zeros_like(element.estimator.coef_)
    element.estimator.intercept_ = np.array([30.0, 40.0, 0.0])
    for calibrator in element.calibrators:
        calibrator.a_, calibrator.b_ = np.float64(-1.0), np.float64(0.0)
    clf = fse.Classifier.from_sklearn(encoder, sk)
    texts = SENTENCES[:40]
    assert clf.predict(texts).tolist() == [0] * 40
    assert ref.labels(clf.decision_function(texts)).tolist() == [1] * 40  # linear@1
    batches = list(clf.predict_batches([texts[:20], texts[20:]], return_vectors=True))
    assert [v for b in batches for v in b.outputs.tolist()] == [0] * 40
    with fse.ShardSink(tmp_path / "vectors", encoder) as sink:
        encoder.encode(texts, workers=1, sink=sink)
    with fse.VectorStore.open(tmp_path / "vectors") as store:
        assert clf.predict(store, workers=2).tolist() == [0] * 40
        proba = clf.predict_proba(store, workers=2)
    assert (proba.argmax(axis=1) == 0).all()


def test_classes_of_other_numpy_types_are_compared_by_value():
    encoder, sk = _fitted(2)
    for classes in (np.array([0, 1], np.uint64), np.array(["no", "yes"], dtype=object)):
        model = copy.deepcopy(sk)
        element = model.calibrated_classifiers_[0]
        for target, name in (
            (model, "classes_"),
            (element, "classes"),
            (element.estimator, "classes_"),
        ):
            setattr(target, name, classes)
        clf = fse.Classifier.from_sklearn(encoder, model)
        assert clf.classes_.tolist() == classes.tolist()
