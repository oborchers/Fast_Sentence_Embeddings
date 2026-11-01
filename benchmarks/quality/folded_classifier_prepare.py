"""Train one raw-mean linear head for both folded and encode-plus-matmul inference."""

from __future__ import annotations

import json
import time

import numpy as np
from compare_artifacts import file_digest, native_encoder_identity
from compare_inputs import identity
from compare_models import MODELS
from compare_runtime import verify_backend_identity
from compare_snapshots import cached_snapshot
from native_classifier import train as train_mlp
from prepare_fineweb import digest, metrics

REPO = "minishlab/potion-base-8M"
MLP_SOURCE = "https://github.com/MinishLab/model2vec/blob/745bf944406a125be439ddfdfe523747baec1983/model2vec/inference/mlp.py"
MLP_DESCRIPTION = (
    "fine-tuned embeddings and MLP, one 512-unit ReLU hidden layer, softmax or sigmoid output; "
    "PyTorch training, exported plain NumPy inference, model2vec 0.10.0"
)


def linear_predict(encoder, weight, bias, classes, rows, workers, *, encode_seconds=None):
    start = time.perf_counter() if encode_seconds is not None else None
    vectors = encoder.encode(rows, workers=workers)
    if encode_seconds is not None:
        encode_seconds.append(time.perf_counter() - start)
    logits = vectors @ weight.T + bias
    indices = (logits[:, 0] > 0).astype(np.intp) if logits.shape[1] == 1 else logits.argmax(axis=1)
    return classes[indices]


def fitted_head(suite, features, labels):
    from prepare_setfit import fit
    from sklearn.linear_model import LogisticRegression

    if suite == "setfit":
        return fit(features, labels)
    return LogisticRegression(max_iter=1000, random_state=0).fit(features, labels)


def write_evaluation(folder, rows, labels):
    directory = folder / "pipeline-inputs"
    directory.mkdir()
    path = directory / "evaluation.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for text, label in zip(rows, labels, strict=True):
            stream.write(json.dumps({"text": text, "label": int(label)}) + "\n")
    return {
        "file": str(path.relative_to(folder)),
        "checksum": file_digest(path),
        "rows": len(rows),
        "input_digest": digest(rows),
    }


def save_linear(encoder, fitted, folder):
    import joblib

    import fse

    model = fse.Classifier.from_sklearn(encoder, fitted, normalized=False)
    if model.normalized:
        raise ValueError("folded classifier must read raw, unnormalized means")
    encoder.save(folder / "raw-encoder.safetensors")
    np.savez(
        folder / "linear-head.npz",
        weight=model.coef_,
        bias=model.intercept_,
        classes=model.classes_,
    )
    # Export the same coefficients through sklearn, rounded once to fse's float32 head.
    fitted.coef_ = model.coef_
    fitted.intercept_ = model.intercept_
    fitted.classes_ = model.classes_
    joblib.dump(fitted, folder / "same-sklearn-head.joblib")
    model.save(folder / "folded-linear.safetensors", standalone=True)
    loaded = fse.Classifier.load(folder / "folded-linear.safetensors", mmap=True)
    if not loaded.standalone or loaded.normalized:
        raise ValueError("saved classifier did not retain its folded raw-mean table")
    return model, loaded


def raw_mean_encoder(repo, revision, *, local_files_only=True):
    import fse

    encoder = fse.StaticModel.from_pretrained(
        repo, revision=revision, normalize=False, local_files_only=local_files_only
    ).encoder
    if encoder.steps or encoder.word_vectors.dtype != "float32":
        raise ValueError("raw-mean arm must have no normalization step and a float32 table")
    return encoder


def prepare(suite, name, data, provenance, folder):
    from threadpoolctl import threadpool_limits

    folder.mkdir()
    train, labels, evaluation, truth = data
    labels = np.asarray(labels, dtype=np.int64)
    runtime = identity("fse")
    snapshot = cached_snapshot(REPO, MODELS[REPO])
    source = native_encoder_identity(REPO, snapshot)
    encoder = raw_mean_encoder(REPO, MODELS[REPO])
    with threadpool_limits(limits=2):
        start = time.monotonic()
        fitted = fitted_head(suite, encoder.encode(train, workers=2), labels)
        training_seconds = time.monotonic() - start
        original, folded = save_linear(encoder, fitted, folder)
        folded_prediction = folded.predict(evaluation, workers=2)
        if not np.array_equal(folded_prediction, original.predict(evaluation, workers=2)):
            raise ValueError(
                "standalone storage changed the internally folded algorithm's predictions"
            )
        explicit_prediction = linear_predict(
            encoder, original.coef_, original.intercept_, original.classes_, evaluation, 2
        )
    verify_backend_identity("fse", runtime)
    record = {
        **provenance,
        **runtime,
        "suite": suite,
        "dataset_name": name,
        "model": REPO,
        "revision": MODELS[REPO],
        "base_encoder": source,
        "train_rows": len(train),
        "train_digest": digest(train),
        "evaluation_rows": len(evaluation),
        "evaluation": write_evaluation(folder, evaluation, truth),
        "training_feature_space": "raw unnormalized means, StaticModel.from_pretrained(normalize=False).encoder",
        "raw_encoder_fingerprint": encoder.fingerprint,
        "folded_classifier_fingerprint": folded.fingerprint,
        "linear_head": {
            "kind": "linear",
            "outputs": folded.n_outputs,
            "original_embedding_dim": encoder.output_dim,
            "parameters": fitted.get_params(),
        },
        "folded_storage": "standalone table has one float32 value per logit per token; same algorithm as automatic raw-head folding",
        "training_tool_chain": "sklearn, frozen embeddings and linear head",
        "training_seconds": training_seconds,
        "training_seconds_scope": "raw feature encoding and sklearn fit, separate from inference",
        "quality": {
            "folded": metrics(truth, folded_prediction),
            "encode_matmul": metrics(truth, explicit_prediction),
            "label_disagreements": int(np.count_nonzero(folded_prediction != explicit_prediction)),
        },
        "rounding": "same float32 coefficients and bias; folding and explicit matmul can differ in accumulation order",
        "files": {
            p.name: {"bytes": p.stat().st_size, "checksum": file_digest(p)}
            for p in folder.iterdir()
            if p.is_file()
        },
    }
    (folder / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def prepare_mlp(data, provenance, folder, record):
    mlp = train_mlp(data, REPO, folder / "model2vec-mlp", provenance, timed_training=True)
    record["mlp"] = {
        "description": MLP_DESCRIPTION,
        "source": MLP_SOURCE,
        "quality": {key: mlp[key] for key in ("accuracy", "macro_f1")},
        "training_record": mlp,
    }
    (folder / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
