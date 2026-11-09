"""Read the proven standardized matrix through short-lived bounded windows."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from inputs import load, read, write

CONFIG = Path(__file__).with_name("sgd.json")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def settings(parent):
    config = {**load(parent / "config.json"), **read(CONFIG)}
    if version("scikit-learn") != config["sklearn_version"]:
        raise ValueError("SGD requires the pinned and functionally checked sklearn version")
    return config


def inspect_arrays(source, config):
    x = np.load(source / "train.npy", mmap_mode="r")
    y = np.load(source / "targets.npy", mmap_mode="r")
    expected = (config["train_rows"], config["dimension"])
    if x.shape != expected or x.dtype != np.float32:
        raise ValueError("expected the full standardized float32 matrix")
    if y.shape != (expected[0], len(config["labels"])) or y.dtype != np.bool_:
        raise ValueError("expected the matching six binary training columns")
    del x, y
    return expected


def inspect_scaler(source, expected):
    scaler = joblib.load(source / "scaler.joblib")
    if not (scaler.with_mean and scaler.with_std) or scaler.n_features_in_ != expected[1]:
        raise ValueError("source scaler does not match the standardized matrix")
    if not np.all(np.asarray(scaler.n_samples_seen_) == expected[0]):
        raise ValueError("source scaler was not fitted on the full training split")
    if (
        not np.isfinite(scaler.mean_).all()
        or not (np.isfinite(scaler.scale_) & (scaler.scale_ > 0)).all()
    ):
        raise ValueError("source scaler has invalid statistics")
    # The source writer saves the scaler only after every in-place transform and flush.
    if (source / "scaler.joblib").stat().st_mtime_ns < (source / "train.npy").stat().st_mtime_ns:
        raise ValueError("source matrix changed after its completed standardization")
    return scaler


def inspect_source(parent, source, config):
    if not read(parent / "test-hold.json")["active"]:
        raise ValueError("the official test must remain held")
    coverage = read(source / "coverage.json")
    if coverage != read(parent / "selection.json")["full_train"]:
        raise ValueError("source coverage differs from the pinned full training split")
    source_config = read(source / "config.json")
    for key in ("dataset", "encoders", "labels", "target_threshold", "train_rows", "dimension"):
        if source_config[key] != config[key]:
            raise ValueError("source training provenance differs from the pinned SGD input")
    expected = inspect_arrays(source, config)
    scaler = inspect_scaler(source, expected)
    positive = np.asarray(coverage["positive"], dtype=np.int64)
    negative = np.asarray(coverage["negative"], dtype=np.int64)
    if ((positive <= 0) | (negative <= 0)).any():
        raise ValueError("global class weights require both training classes")
    weights = [
        {0: expected[0] / (2 * int(neg)), 1: expected[0] / (2 * int(pos))}
        for pos, neg in zip(positive, negative, strict=True)
    ]
    receipt = {
        "source": str(source),
        "shape": list(expected),
        "coverage": coverage,
        "scaler_sha256": digest(source / "scaler.joblib"),
        "source_config_sha256": digest(source / "config.json"),
        "class_weights": [{str(k): v for k, v in row.items()} for row in weights],
        "standardized_input": True,
        "second_scaling": False,
    }
    return scaler, weights, receipt


def window(source, start, stop):
    mapped = np.load(source / "train.npy", mmap_mode="r")
    x = np.array(mapped[start:stop], dtype=np.float32, order="C", copy=True)
    del mapped
    mapped = np.load(source / "targets.npy", mmap_mode="r")
    y = np.array(mapped[start:stop], dtype=np.int64, copy=True)
    del mapped
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("invalid SGD training window")
    return x, y


def batches(source, config, epoch, rows):
    rng = np.random.default_rng(config["seed"] + epoch)
    width = config["batch_rows"]
    for block in rng.permutation((rows + width - 1) // width):
        first = int(block) * width
        x, y = window(source, first, min(first + width, rows))
        order = rng.permutation(len(x))
        yield x[order], y[order]


def receipt(path, value):
    temporary = path.with_name(path.name + ".pending")
    write(temporary, value)
    temporary.replace(path)
