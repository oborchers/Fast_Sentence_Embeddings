"""Reconstruct the disclosed MinishLab frozen-encoder classification protocol.

The pinned historical README says 14 datasets but displays 15 rows. This script
uses all 15 displayed rows, pinned by the source manifest. Original hardware,
training selection, CV settings and averaging for 17,925 samples/s are unknown.
Consequently this is a reconstruction, not an exact historical replication.
Use exactly 1,000 seed-0 stratified training examples and every test example.
LogisticRegressionCV uses five folds, ten Cs, max_iter=1000 and random_state=0.
The TF-IDF vocabulary and IDF weights are fitted only on the selected train set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_inputs import identity  # isort: skip
from compare_runtime import backend_identity, verify_backend_identity  # isort: skip
from compare_artifacts import (
    check_head_encoder,
    checked_encoder,
    native_encoder_identity,
    record_head_encoder,
    seal_preparation,
)  # isort: skip

from compare_models import MODELS, artifact_name  # isort: skip
from quality_coordination import task_reservation  # isort: skip
from prepare_fineweb import digest, metrics  # isort: skip
from classification_training import fitted as training_fit  # isort: skip

SOURCES = Path(__file__).resolve().parents[1] / "baseline/dr151_setfit_sources_20261101.json"


def dataset(specification: dict, cache: Path, *, download: bool, admission=None):
    from huggingface_hub import snapshot_download
    from sklearn.model_selection import StratifiedShuffleSplit

    folder = Path(
        snapshot_download(
            specification["dataset"],
            revision=specification["revision"],
            repo_type="dataset",
            cache_dir=cache / "hub",
            allow_patterns=["train.jsonl", "test.jsonl"],
            local_files_only=not download,
            max_workers=2,
        )
    )
    if admission is not None:
        size = sum((folder / f"{split}.jsonl").stat().st_size for split in ("train", "test"))
        admission(size * 32, "SetFit decoded input and temporary JSON records")
    train, evaluation = (
        [
            json.loads(line)
            for line in (folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for split in ("train", "test")
    )
    if len(train) <= 1000:
        raise ValueError("reconstruction needs more than 1000 training examples")
    labels = np.array([row["label"] for row in train])
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=1000, random_state=0)
    chosen, _ = next(splitter.split(np.zeros(len(train)), labels))
    return (
        [train[i]["text"] for i in chosen],
        labels[chosen],
        [row["text"] for row in evaluation],
        np.array([row["label"] for row in evaluation]),
    )


def fit(features, labels):
    from sklearn.linear_model import LogisticRegressionCV

    return LogisticRegressionCV(Cs=10, cv=5, max_iter=1000, random_state=0).fit(features, labels)


def save_result(args, provenance, backend, model, labels, predicted, **extra):
    verify_backend_identity(backend, provenance)
    row = {
        **provenance,
        **metrics(labels, predicted),
        "backend": backend,
        "model": model,
        **extra,
    }
    with args.results.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    print(
        json.dumps({key: row[key] for key in ("dataset", "backend", "model", "macro_f1")}),
        flush=True,
    )


def heads(args, data, provenance, repo, output):
    import joblib
    from compare_snapshots import cached_snapshot
    from model2vec import StaticModel

    from fse import Classifier, Encoder

    provenance = {**provenance, **backend_identity("fse")}
    train, y_train, evaluation, y_eval = data
    encoder = Encoder.load(args.models / artifact_name(repo, "float32"))
    training_source = checked_encoder(encoder, args.models, repo, "float32")
    fitted, training = training_fit(
        args,
        lambda: fit(encoder.encode(train, workers=2), y_train),
        "sklearn LogisticRegressionCV and frozen fse encoder, includes train encoding",
    )
    stem = repo.rsplit("/", maxsplit=1)[-1]
    joblib.dump(fitted, output / f"{stem}-fse-sklearn.joblib")
    record_head_encoder(output, f"{stem}-fse-sklearn.joblib", training_source)
    reference = None
    for dtype in ("float32", "float16", "int8"):
        enc = Encoder.load(args.models / artifact_name(repo, dtype))
        source = checked_encoder(enc, args.models, repo, dtype)
        check_head_encoder({"head_encoders": {"training": training_source}}, "training", source)
        model = Classifier.from_sklearn(enc, fitted)
        predicted = model.predict(evaluation, workers=2)
        if reference is None:
            reference = predicted
        save_result(
            args,
            provenance,
            "fse",
            repo,
            y_eval,
            predicted,
            table_dtype=dtype,
            revision=MODELS[repo],
            head=fitted.get_params(),
            selected_C=fitted.C_.tolist(),
            encoder_fingerprint=enc.fingerprint,
            head_training_table_dtype="float32",
            prediction_disagreement_vs_float32=float(np.mean(predicted != reference)),
            **training,
        )
    provenance = {**provenance, **backend_identity("model2vec")}
    native_folder = cached_snapshot(repo, MODELS[repo])
    native = StaticModel.from_pretrained(native_folder)
    source = native_encoder_identity(repo, native_folder)
    fitted, training = training_fit(
        args,
        lambda: fit(native.encode(train, use_multiprocessing=False), y_train),
        "sklearn LogisticRegressionCV and frozen model2vec encoder, includes train encoding",
    )
    predicted = fitted.predict(native.encode(evaluation, use_multiprocessing=False))
    joblib.dump(fitted, output / f"{stem}-model2vec-sklearn.joblib")
    record_head_encoder(output, f"{stem}-model2vec-sklearn.joblib", source)
    save_result(
        args,
        provenance,
        "model2vec",
        repo,
        y_eval,
        predicted,
        table_dtype="float32",
        revision=MODELS[repo],
        head=fitted.get_params(),
        selected_C=fitted.C_.tolist(),
        native_multiprocessing=False,
        **training,
    )


def tfidf(args, data, provenance, output):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline

    provenance = {**provenance, **backend_identity("tfidf")}
    train, y_train, evaluation, y_eval = data
    search = GridSearchCV(
        make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=0)),
        {"logisticregression__C": np.logspace(-4, 4, 10).tolist()},
        cv=5,
        n_jobs=1,
    ).fit(train, y_train)
    pipeline = search.best_estimator_
    vectorizer = pipeline.named_steps["tfidfvectorizer"]
    head = pipeline.named_steps["logisticregression"]
    predicted = pipeline.predict(evaluation)
    joblib.dump(pipeline, output / "tfidf-sklearn.joblib")
    save_result(
        args,
        provenance,
        "tfidf",
        "TfidfVectorizer + LogisticRegression, pipeline GridSearchCV",
        y_eval,
        predicted,
        vocabulary_rows=len(vectorizer.vocabulary_),
        head=head.get_params(),
        selected_C=search.best_params_["logisticregression__C"],
        vectorizer="scikit-learn defaults",
    )


def prepare(args, specification):
    from threadpoolctl import threadpool_limits

    stem = specification["dataset"].removeprefix("SetFit/")
    output = args.output / stem
    output.mkdir()  # Never reuse a partially trained classifier directory.
    with task_reservation(args.coordination), threadpool_limits(limits=2):
        data = dataset(specification, args.cache, download=True)
        provenance = {
            **identity(),
            "dataset": specification["dataset"],
            "dataset_revision": specification["revision"],
            "train_rows": len(data[0]),
            "evaluation_rows": len(data[2]),
            "train_digest": digest(data[0]),
            "evaluation_digest": digest(data[2]),
            "split": "1000 stratified, seed 0",
            "protocol": "historical 15 displayed rows reconstruction",
            "workers": 2,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
        }
        (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for repo in ("minishlab/potion-base-8M", "minishlab/potion-base-32M"):
            heads(args, data, provenance, repo, output)
        tfidf(args, data, provenance, output)
        seal_preparation(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/fse-quality/setfit")
    parser.add_argument("--models", type=Path, default=Path.home() / ".cache/fse-quality/models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--pod-training-times", action="store_true")
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("run with nice -n 10")
    args.output.mkdir()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = "2"
    for specification in json.loads(SOURCES.read_text())["datasets"]:
        prepare(args, specification)


if __name__ == "__main__":
    main()
