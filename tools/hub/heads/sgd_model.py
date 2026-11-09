"""Independent online sklearn models and an explicit affine multilabel export."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

import fse


def create(config, weights, c):
    alpha = 1 / (c * config["train_rows"])
    return [
        SGDClassifier(
            loss=config["loss"],
            penalty=config["penalty"],
            alpha=alpha,
            learning_rate=config["learning_rate"],
            eta0=config["eta0"],
            power_t=config["power_t"],
            average=config["average"],
            early_stopping=False,
            tol=None,
            shuffle=False,
            random_state=config["seed"],
            n_jobs=1,
            class_weight=weight,
        )
        for weight in weights
    ]


def update(models, x, y, budget):
    for label, model in enumerate(models):
        budget.check()
        model.partial_fit(x, y[:, label], classes=np.array([0, 1], dtype=np.int64))
        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
            raise FloatingPointError("SGD candidate has nonfinite coefficients")


class Reference:
    """Actual independent sklearn outputs, without pretending to be an sklearn OVR wrapper."""

    def __init__(self, scaler, models):
        self.scaler, self.models = scaler, models

    def predict_proba(self, x):
        scaled = self.scaler.transform(x, copy=True)
        return np.column_stack([model.predict_proba(scaled)[:, 1] for model in self.models])

    def predict(self, x):
        scaled = self.scaler.transform(x, copy=True)
        return np.column_stack([model.predict(scaled) for model in self.models])


def export(encoder, scaler, models, labels):
    coef = np.concatenate([model.coef_.astype(np.float64) for model in models])
    intercept = np.concatenate([model.intercept_.astype(np.float64) for model in models])
    a64 = coef / np.asarray(scaler.scale_, dtype=np.float64)
    b64 = intercept - a64 @ np.asarray(scaler.mean_, dtype=np.float64)
    model = fse.Classifier.from_arrays(encoder, a64, b64, labels=labels, link="multilabel")
    return model, Reference(scaler, models), (a64, b64)
