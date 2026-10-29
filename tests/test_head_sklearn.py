"""Heads read from fitted scikit-learn estimators (fse._head._sklearn), compared with
scikit-learn's own predictions within the error bound of tests/support/head_bound.py."""

import warnings

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn import linear_model, pipeline, preprocessing, svm  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402

import fse  # noqa: E402
from fse._head import _sklearn as reader  # noqa: E402
from support import head as support  # noqa: E402
from support import head_bound  # noqa: E402
from support.encoder import finished  # noqa: E402
from support.runtime import texts  # noqa: E402

F32 = np.float32
SENTENCES = texts(700, seed=8)


def _data(encoder, classes=2, seed=0):
    x = encoder.encode(SENTENCES)
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((encoder.output_dim, classes))
    score = x.astype(np.float64) @ hidden + 0.3 * rng.standard_normal((len(x), classes))
    y = (score[:, 0] > 0).astype(int) if classes == 2 else np.argmax(score, axis=1)
    return x, y


def _fit(model, x, y):
    with warnings.catch_warnings():
        # scikit-learn's own deprecations and convergence notes, which vary by version.
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", sklearn.exceptions.ConvergenceWarning)
        # Old scikit-learn (1.5) passes options to newer SciPy's L-BFGS-B that it deprecates.
        warnings.filterwarnings(
            "ignore", "scipy.optimize: The `disp` and `iprint`", DeprecationWarning
        )
        return model.fit(x, y)


def _exact(clf, encoder):
    """The spec, fse's vectors and the float64 head scikit-learn applies to them."""
    spec = encoder._spec
    x = encoder.encode(SENTENCES)
    found = reader.read(clf, "classification" if hasattr(clf, "classes_") else "regression")
    a64, b64 = found.weight, found.bias
    return spec, x, a64, b64


def _check(model, sk, encoder):
    """fse's labels and probabilities against the exact float64 head: labels differ only
    where the bound allows, probabilities within it; scikit-learn's own output is that head
    up to float64 rounding."""
    spec, x, a64, b64 = _exact(sk, encoder)
    z_star = x.astype(np.float64) @ a64.T + b64
    a, b = model.coef_, model.intercept_
    if model.normalized:
        z = support.logits(spec, a, b, SENTENCES, normalized=True)
        bound = head_bound.unfolded(x, a64, b64, a, b, z)
    else:
        z = support.logits(spec, a, b, SENTENCES)
        table = support.folded_pool(spec.pool_input(), a)
        batch = support.reference_batch(spec, SENTENCES)

        def rows(kind, i):
            return table[0][i] if kind == "word" else table[3][i]

        bound = head_bound.folded(spec, batch, rows, x, a64, b64, a, b, z)
    assert (np.abs(z - z_star) <= bound).all()
    labels = model.predict(SENTENCES)
    want = sk.predict(x)
    changed = labels != want
    assert not (changed & ~head_bound.allowed_label_changes(z_star, bound)).any()
    decision = sk.decision_function(x) if hasattr(sk, "decision_function") else sk.predict(x)
    decision = decision.reshape(len(x), -1)
    if decision.shape[1] == z_star.shape[1]:  # scikit-learn's head, in float32 on float32 input
        assert (np.abs(decision - z_star) <= _float32_error(x, a64, b64)).all()
    return z, z_star, bound


def _float32_error(x, a64, b64):
    """How far a float32 product of any order (scikit-learn keeps the dtype of its input)
    may be from the float64 head."""
    terms = np.abs(x.astype(np.float64)) @ np.abs(a64).T + np.abs(b64)
    return head_bound.gamma(x.shape[1] + 4) * terms + 1e-30


BINARY = [
    linear_model.LogisticRegression(),
    linear_model.LogisticRegression(fit_intercept=False),
    linear_model.LogisticRegression(solver="liblinear"),
    linear_model.SGDClassifier(loss="log_loss", random_state=0),
    linear_model.SGDClassifier(loss="hinge", random_state=0),
    svm.LinearSVC(),
    linear_model.RidgeClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.Perceptron(random_state=0),
]


@pytest.mark.parametrize("sk", BINARY, ids=lambda m: f"{type(m).__name__}-{len(m.get_params())}")
@pytest.mark.parametrize("normalize", [False, True])
def test_binary_classifiers(sk, normalize):
    encoder = finished(normalize=normalize)
    x, y = _data(encoder)
    sk = _fit(sk, x, y)
    model = fse.Classifier.from_sklearn(encoder, sk)
    assert model.normalized == normalize
    assert model.n_outputs == 1
    assert model.classes_.tolist() == sk.classes_.tolist()
    _z, _z_star, bound = _check(model, sk, encoder)
    if hasattr(sk, "predict_proba"):
        assert model.link == "sigmoid@1"
        got = model.predict_proba(SENTENCES)
        want = sk.predict_proba(x)
        tolerance = head_bound.proba_tolerance("sigmoid@1", bound, 1)
        assert (np.abs(got - want).max(axis=1) <= tolerance).all()
    else:
        assert model.link == "none"


MULTI = [
    (linear_model.LogisticRegression(), "softmax@1"),
    (linear_model.LogisticRegression(solver="liblinear"), None),  # ovr before 1.8
    (linear_model.SGDClassifier(loss="log_loss", random_state=0), "ovr@1"),
    (svm.LinearSVC(), "none"),
    (linear_model.RidgeClassifier(), "none"),
    (linear_model.Perceptron(random_state=0), "none"),
]


@pytest.mark.parametrize(
    ("sk", "link"),
    MULTI,
    ids=lambda m: type(m).__name__ if not isinstance(m, str | None) else str(m),
)
def test_multiclass_classifiers(sk, link):
    encoder = finished(normalize=False)
    x, y = _data(encoder, classes=4)
    try:
        sk = _fit(sk, x, y)
    except ValueError as error:  # liblinear takes two classes only from scikit-learn 1.8 on
        pytest.skip(str(error))
    model = fse.Classifier.from_sklearn(encoder, sk)
    assert model.n_outputs == 4
    if link is not None:
        assert model.link == link
    _, _z_star, bound = _check(model, sk, encoder)
    if model.link == "softmax@1":
        tolerance = head_bound.proba_tolerance("softmax@1", bound, 4)
        got = model.predict_proba(SENTENCES)
        assert (np.abs(got - sk.predict_proba(x)).max(axis=1) <= tolerance).all()
    if model.link == "ovr@1":
        got = model.predict_proba(SENTENCES)
        np.testing.assert_allclose(got, sk.predict_proba(x), atol=1e-5)


def test_the_rule_of_older_versions_with_multi_class():
    """scikit-learn 1.5 to 1.7 kept multi_class: a binary multinomial model has one logit
    z and the probabilities softmax of [-z, z]; ovr gives sigmoid, or ovr for more classes."""
    encoder = finished(normalize=False)
    x, y = _data(encoder)
    sk = _fit(linear_model.LogisticRegression(), x, y)
    for multi_class, want in [
        ("multinomial", ("softmax@1", 1)),
        ("ovr", ("sigmoid@1", 1)),
        ("auto", ("sigmoid@1", 1)),
    ]:
        sk.multi_class = multi_class
        model = fse.Classifier.from_sklearn(encoder, sk)
        assert (model.link, model.n_outputs) == want
    sk.multi_class = "multinomial"
    model = fse.Classifier.from_sklearn(encoder, sk)
    z = sk.decision_function(x)
    decision = model.decision_function(SENTENCES)
    assert decision.shape == z.shape == (len(SENTENCES),)  # as scikit-learn gives it
    np.testing.assert_allclose(decision, z, rtol=1e-4, atol=1e-5)
    exact = 1 / (1 + np.exp(-2 * z))  # softmax of [-z, z]
    assert np.abs(model.predict_proba(SENTENCES)[:, 1] - exact).max() < 1e-5
    changed = model.predict(SENTENCES) != sk.classes_[(z > 0).astype(int)]
    assert (np.abs(z[changed]) < 1e-5).all()
    x3, y3 = _data(encoder, classes=3)
    sk3 = _fit(linear_model.LogisticRegression(), x3, y3)
    sk3.solver, sk3.multi_class = "liblinear", "auto"  # as 1.5 to 1.7 fitted it, with ovr
    assert fse.Classifier.from_sklearn(encoder, sk3).link == "ovr@1"
    del sk3.multi_class
    assert fse.Classifier.from_sklearn(encoder, sk3).link == "softmax@1"  # 1.8: softmax


REGRESSORS = [
    linear_model.LinearRegression(),
    linear_model.Ridge(),
    linear_model.RidgeCV(),
    linear_model.Lasso(alpha=0.01),
    linear_model.LassoCV(cv=2),
    linear_model.ElasticNet(alpha=0.01),
    linear_model.ElasticNetCV(cv=2),
    linear_model.SGDRegressor(random_state=0),
    svm.LinearSVR(random_state=0),
    linear_model.HuberRegressor(),
]


@pytest.mark.parametrize("sk", REGRESSORS, ids=lambda m: type(m).__name__)
def test_regressors(sk):
    encoder = finished(normalize=False)
    x, _ = _data(encoder)
    y = x.astype(np.float64) @ np.arange(1, 7) + 0.5
    sk = _fit(sk, x, y)
    model = fse.Regressor.from_sklearn(encoder, sk)
    spec, x, a64, b64 = _exact(sk, encoder)
    z = model.predict(SENTENCES)[:, None]
    z_star = x.astype(np.float64) @ a64.T + b64
    table = support.folded_pool(spec.pool_input(), model.coef_)
    batch = support.reference_batch(spec, SENTENCES)
    bound = head_bound.folded(
        spec, batch, lambda _kind, i: table[0][i], x, a64, b64, model.coef_, model.intercept_, z
    )
    assert (np.abs(z - z_star) <= bound).all()
    assert (np.abs(sk.predict(x) - z_star[:, 0]) <= _float32_error(x, a64, b64)[:, 0]).all()


def test_two_targets():
    encoder = finished(normalize=False)
    x, _ = _data(encoder)
    y = np.stack([x[:, 0] * 2.0, x[:, 1] - 1.0], axis=1)
    sk = linear_model.Ridge().fit(x, y)
    model = fse.Regressor.from_sklearn(encoder, sk)
    assert model.predict(SENTENCES).shape == (len(SENTENCES), 2)
    np.testing.assert_allclose(model.predict(SENTENCES), sk.predict(x), atol=1e-5)


def test_pipelines_with_a_normalizer_and_a_scaler():
    encoder = finished(normalize=False)
    x, y = _data(encoder)
    scaled = _fit(
        pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression()),
        x,
        y,
    )
    model = fse.Classifier.from_sklearn(encoder, scaled)
    assert not model.normalized  # the scaler folds into A and b; the head still folds
    np.testing.assert_allclose(
        model.decision_function(SENTENCES), scaled.decision_function(x), atol=1e-4
    )
    both = pipeline.make_pipeline(
        preprocessing.Normalizer(),
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(),
    )
    both = _fit(both, x, y)
    model = fse.Classifier.from_sklearn(encoder, both)
    assert model.normalized  # a Normalizer first: the head reads normalized vectors
    np.testing.assert_allclose(
        model.decision_function(SENTENCES), both.decision_function(x), atol=1e-4
    )
    with pytest.raises(ValueError, match="normalized=False contradicts it"):
        fse.Classifier.from_sklearn(encoder, both, normalized=False)
    passthrough = _fit(
        pipeline.Pipeline([("skip", "passthrough"), ("lr", linear_model.LogisticRegression())]),
        x,
        y,
    )
    assert fse.Classifier.from_sklearn(encoder, passthrough).n_outputs == 1


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda: pipeline.make_pipeline(
                preprocessing.StandardScaler(),
                preprocessing.Normalizer(),
                linear_model.LogisticRegression(),
            ),
            "only an optional Normalizer",
        ),
        (
            lambda: pipeline.make_pipeline(
                preprocessing.MinMaxScaler(), linear_model.LogisticRegression()
            ),
            "holds MinMaxScaler",
        ),
        (
            lambda: pipeline.make_pipeline(
                preprocessing.Normalizer(norm="l1"), linear_model.LogisticRegression()
            ),
            "norm='l2'",
        ),
        (
            lambda: OneVsRestClassifier(linear_model.LogisticRegression()),
            "not a scikit-learn classification head",
        ),
        (lambda: linear_model.LinearRegression(), "not a scikit-learn classification head"),
        (
            lambda: linear_model.SGDClassifier(loss="modified_huber", random_state=0),
            "modified_huber",
        ),
    ],
)
def test_refusals(make, match):
    encoder = finished(normalize=False)
    x, y = _data(encoder)
    sk = _fit(make(), x, y)
    with pytest.raises(ValueError, match=match):
        fse.Classifier.from_sklearn(encoder, sk)


def test_more_refusals():
    encoder = finished(normalize=False)
    x, y = _data(encoder)
    with pytest.raises(ValueError, match="has no coef_; is it fitted"):
        fse.Classifier.from_sklearn(encoder, linear_model.LogisticRegression())
    sparse = _fit(linear_model.SGDClassifier(random_state=0), x, y).sparsify()
    with pytest.raises(ValueError, match="not a dense array"):
        fse.Classifier.from_sklearn(encoder, sparse)
    wide = _fit(linear_model.LogisticRegression(), np.hstack([x, x]), y)
    with pytest.raises(ValueError, match=r"shape \(k, 6\)"):
        fse.Classifier.from_sklearn(encoder, wide)
    with pytest.raises(ValueError, match="not a scikit-learn regression head"):
        fse.Regressor.from_sklearn(encoder, _fit(linear_model.LogisticRegression(), x, y))
    labels = _fit(linear_model.LogisticRegression(), x, np.where(y == 1, "yes", "no"))
    assert fse.Classifier.from_sklearn(encoder, labels).classes_.tolist() == ["no", "yes"]


def test_a_head_without_probabilities_names_its_estimator():
    encoder = finished(normalize=False)
    x, y = _data(encoder)
    model = fse.Classifier.from_sklearn(encoder, _fit(svm.LinearSVC(), x, y))
    with pytest.raises(ValueError, match=r"this classifier \(LinearSVC\) has no probabilities"):
        model.predict_proba(SENTENCES)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_every_kind_of_standard_scaler(with_mean, with_std):
    """mean_ stays set without centering (scikit-learn keeps it for the variance): only the
    steps the scaler applies may fold into the head."""
    encoder = finished(normalize=False)
    x, y = _data(encoder)
    scaler = preprocessing.StandardScaler(with_mean=with_mean, with_std=with_std)
    model = _fit(pipeline.make_pipeline(scaler, linear_model.LogisticRegression()), x, y)
    clf = fse.Classifier.from_sklearn(encoder, model)
    want = model.decision_function(encoder.encode(SENTENCES))
    np.testing.assert_allclose(clf.decision_function(SENTENCES), want, rtol=1e-4, atol=1e-4)


def test_newton_cholesky_with_many_classes_has_the_link_of_its_version():
    encoder = finished(normalize=False)
    x, y = _data(encoder, classes=3)
    try:
        sk = _fit(linear_model.LogisticRegression(solver="newton-cholesky"), x, y)
    except ValueError as error:  # versions without it for several classes
        pytest.skip(str(error))
    clf = fse.Classifier.from_sklearn(encoder, sk)
    from fse._head._sklearn import installed_version

    ovr = getattr(sk, "multi_class", None) in ("auto", "deprecated") and installed_version() < (
        1,
        6,
    )
    assert clf.link == ("ovr@1" if ovr else "softmax@1")
    np.testing.assert_allclose(clf.predict_proba(SENTENCES), sk.predict_proba(x), atol=1e-5)


def test_a_fitted_binary_multinomial_model_where_the_version_has_one():
    encoder = finished(normalize=False)
    x, y = _data(encoder)
    try:
        sk = _fit(linear_model.LogisticRegression(multi_class="multinomial"), x, y)
    except TypeError as error:  # multi_class was removed in 1.8
        pytest.skip(str(error))
    clf = fse.Classifier.from_sklearn(encoder, sk)
    assert (clf.link, clf.n_outputs) == ("softmax@1", 1)
    np.testing.assert_allclose(clf.predict_proba(SENTENCES), sk.predict_proba(x), atol=1e-5)
    got = clf.decision_function(SENTENCES)
    assert got.shape == sk.decision_function(x).shape
    np.testing.assert_allclose(got, sk.decision_function(x), rtol=1e-4, atol=1e-5)


def test_one_target_of_two_dimensions_keeps_its_shape(tmp_path):
    encoder = finished(normalize=False)
    x, _ = _data(encoder)
    y = (x.astype(np.float64) @ np.arange(1, 7))[:, None]
    for target, shape in ((y, (len(x), 1)), (y[:, 0], (len(x),))):
        sk = linear_model.LinearRegression().fit(x, target)
        model = fse.Regressor.from_sklearn(encoder, sk)
        assert sk.predict(x).shape == shape
        assert model.predict(SENTENCES).shape == shape
        model.save(tmp_path / f"{len(shape)}.safetensors")
        assert (
            fse.Regressor.load(tmp_path / f"{len(shape)}.safetensors").predict(SENTENCES).shape
            == shape
        )


@pytest.mark.parametrize(
    ("version", "link"), [((1, 5), "ovr@1"), ((1, 6), "softmax@1"), ((1, 7), "softmax@1")]
)
def test_newton_cholesky_follows_the_rule_of_each_version(monkeypatch, version, link):
    """Before 1.6 predict_proba took one-vs-rest for newton-cholesky under multi_class auto,
    from 1.6 (multinomial loss, scikit-learn PR 28840) softmax; 1.8 removed multi_class."""
    from fse._head import _sklearn as module

    encoder = finished(normalize=False)
    x, y = _data(encoder, classes=3)
    sk = _fit(linear_model.LogisticRegression(), x, y)
    sk.solver, sk.multi_class = "newton-cholesky", "auto"
    monkeypatch.setattr(module, "installed_version", lambda: version)
    assert fse.Classifier.from_sklearn(encoder, sk).link == link


def test_the_installed_version_is_read_without_importing_scikit_learn():
    from importlib.metadata import version

    from fse._head._sklearn import installed_version

    assert installed_version() == tuple(int(p) for p in version("scikit-learn").split(".")[:2])
