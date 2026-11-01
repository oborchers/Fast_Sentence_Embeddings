"""Model2Vec 0.10.0 training and its documented NumPy inference export.

Training fine-tunes the embedding table and a hidden-layer classifier. This is
a different tool chain and architecture from frozen embeddings with sklearn.
Evaluation receives the same selected train rows and complete test rows as the
frozen-POTION comparisons. No evaluation row enters training or early stopping.
"""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from compare_artifacts import file_digest, native_encoder_identity
from compare_inputs import identity
from compare_models import MODELS
from compare_runtime import verify_backend_identity
from compare_snapshots import cached_snapshot
from prepare_fineweb import digest, metrics

VERSION = "0.10.0"
FIT = {
    "learning_rate": 0.001,
    "batch_size": None,
    "min_epochs": None,
    "max_epochs": -1,
    "early_stopping_patience": 5,
    "test_size": 0.1,
    "device": "cpu",
    "random_seed": 0,
    "token_dropout": 0.0,
    "focal_gamma": 0.0,
}


def artifact_files(folder: Path) -> dict:
    return {
        str(path.relative_to(folder)): {
            "file_bytes": path.stat().st_size,
            "checksum": file_digest(path),
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != "training-record.json"
    }


def checked_pipeline(folder: Path):
    from model2vec.inference import StaticModelPipeline

    record = json.loads((folder / "training-record.json").read_text())
    if artifact_files(folder) != record["export_files"]:
        raise ValueError("native classifier export differs from its fitted artifact record")
    return StaticModelPipeline.from_pretrained(folder), record


def train(data, repo: str, output: Path, provenance: dict, *, timed_training=False) -> dict:
    import torch
    from model2vec.train import StaticModelForClassification

    from fse._container._hash import checksum

    if version("model2vec") != VERSION:
        raise ValueError("native classifier protocol requires model2vec 0.10.0")
    train_rows, train_labels, test_rows, test_labels = data
    output.mkdir()  # Never mix a completed export with a partial training run.
    runtime = identity("model2vec")
    source = cached_snapshot(repo, MODELS[repo])
    base_identity = native_encoder_identity(repo, source)
    torch.manual_seed(0)
    np.random.seed(0)
    start = time.monotonic() if timed_training else None
    classifier = StaticModelForClassification.from_pretrained(
        path=source, n_layers=1, hidden_dim=512, freeze=False, normalize=True
    )
    classifier.fit(train_rows, np.asarray(train_labels, dtype=np.int64), **FIT)
    training_seconds = time.monotonic() - start if start is not None else None
    pipeline = classifier.to_pipeline()
    pipeline.save_pretrained(output)
    # Evaluate the serialized inference export, not the training object.
    from model2vec.inference import StaticModelPipeline

    exported = StaticModelPipeline.from_pretrained(output)
    predicted = exported.predict(test_rows, use_multiprocessing=False)
    verify_backend_identity("model2vec", runtime)
    record = {
        **provenance,
        **runtime,
        **metrics(test_labels, predicted),
        "backend": "model2vec",
        "classifier_kind": "StaticModelForClassification.to_pipeline",
        "model": repo,
        "revision": MODELS[repo],
        "base_encoder": base_identity,
        "train_rows": len(train_rows),
        "evaluation_rows": len(test_rows),
        "train_digest": digest(train_rows),
        "train_label_digest": checksum(np.ascontiguousarray(train_labels, dtype=np.int64)),
        "evaluation_digest": digest(test_rows),
        "evaluation_label_digest": checksum(np.ascontiguousarray(test_labels, dtype=np.int64)),
        "prediction_digest": checksum(np.ascontiguousarray(predicted)),
        "training_seconds": training_seconds,
        "training_seconds_scope": "from_pretrained and fit, excludes export and inference",
        "training_tool_chain": "torch, trainable embeddings and one hidden layer of 512 units",
        "training_comparison": "different tool chains: sklearn frozen encoder versus torch fine tuning",
        "fit": FIT,
        "early_stopping_data": "seed-0 internal 10% split of training rows only",
        "inference_export": "StaticModelPipeline.save_pretrained, NumPy MLP and StaticModel",
        "export_files": artifact_files(output),
        "platform_check": "pending comparison with the other platform on the same dataset and protocol",
    }
    (output / "training-record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
