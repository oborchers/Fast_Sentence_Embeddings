"""Fitted Platt-calibrated scikit-learn classifiers and the error bounds they are compared
within (the head_bound of linear@1, plus the float64 fold of the calibration and the
normalization of one-vs-rest probabilities).

The vectors given to scikit-learn are fse's float32 vectors converted to float64, so that
scikit-learn computes the float64 head that the bounds name: its float32 paths (SGD
coefficients, the scaler of 1.9.1) are exact in float64 and need no extra term."""

import math
import warnings

import numpy as np

from support import head_bound

U = head_bound.U
U64 = head_bound.U64
TAU = 2.0**-126  # the smallest normal float32, the absolute error of an underflowed exp@1
UNDERFLOW = math.log(TAU)  # about -87.3


def fit(model, x, y):
    """``model.fit(x, y)`` without the deprecations and convergence notes that vary by
    version of scikit-learn."""
    import sklearn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", sklearn.exceptions.ConvergenceWarning)
        warnings.filterwarnings(
            "ignore", "scipy.optimize: The `disp` and `iprint`", DeprecationWarning
        )
        return model.fit(x, y)


def calibrator(model):
    """(a, c) of the sigmoids of a fitted ``CalibratedClassifierCV`` (ensemble=False), in
    float64, one per decision column."""
    element = model.calibrated_classifiers_[0]
    a = np.array([float(c.a_) for c in element.calibrators])
    c = np.array([float(c.b_) for c in element.calibrators])
    return a, c


def platt_distance(r, w64, b64, c, offset=None):
    """How far the float64 head ``w64 r + b64`` that fse rounds is from the ideal
    ``-(a (A_f r + b_f) + c)`` (and from scikit-learn's own float64 evaluation of it): a
    chain of ``d + 6`` roundings of the magnitudes, twice (both evaluations). ``offset``
    (n, k) adds the cancellation of a scaler folded inside, ``|a| |W_f| |mean|``."""
    r = np.abs(np.asarray(r, np.float64))
    scale = r @ np.abs(w64).T + np.abs(b64) + 2 * np.abs(c)
    if offset is not None:
        scale = scale + offset
    return 2 * head_bound.gamma(w64.shape[1] + 6, U64) * scale + 1e-300


def ovr_tolerance(z_star, delta, k):
    """The largest difference per row between fse's ``ovr@1`` probabilities and the exact
    ones, given that every logit is within ``delta`` (n, k) of ``z_star`` (n, k), or None
    for a row in the deep underflow, where fse's uniform answer is checked against its own
    reference bits only.

    ``d/dz log sigmoid(z) = 1 - sigmoid(z) <= 1``, so a logit error d changes sigmoid by at
    most the factor ``e^d - 1``; exp@1 is within 2 ulps (relative 4u), then ``1 + e`` and the
    quotient round: ``rho``. The sum in index order has relative error gamma(k - 1) and the
    division u. A component below ln(2**-126) has an absolute error of 2**-126 instead."""
    z_star = np.asarray(z_star, np.float64)
    worst = np.asarray(delta, np.float64).max(axis=1)
    rho = (1 + 4 * U) * (1 + U) / (1 - 5 * U) - 1
    big_r = np.expm1(worst) + rho
    relative = (1 + big_r) * (1 + head_bound.gamma(k - 1)) * (1 + U) / (1 - big_r) - 1
    sigma = 1 / (1 + np.exp(-z_star))
    exact = sigma / sigma.sum(axis=1, keepdims=True)
    total = sigma.sum(axis=1)
    deep = k * TAU > total / 4
    absolute = k * TAU / total
    out = exact.max(axis=1) * relative + np.where((z_star < UNDERFLOW).any(axis=1), absolute, 0)
    out = np.where(deep, np.nan, out)
    return np.where(np.isnan(out), np.nan, out)
