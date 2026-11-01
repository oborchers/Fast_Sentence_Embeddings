"""Train frozen-POTION linear heads and fastText on all FineWeb-Edu annotations.

This is offline classifier training outside fse's inference engine. Use the
isolated comparison environment and nice -n 10. No timing here is a speed claim.
The fixed seed-0 permutation holds out 46,743 of all 467,424 pages. All models
receive the same whitespace-cleaned pages and binary label (annotation >=3).
POTION retains its pretrained length limits. fastText sees each full page.
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
from compare_models import MODELS, artifact_name  # isort: skip
from quality_coordination import task_reservation  # isort: skip

REPO = "HuggingFaceFW/fineweb-edu-llama3-annotations"
REVISION = "72df4c92fb1b48beceb16016e8f695ec40a6c3a5"
EVAL = 46_743


def digest(rows: list[str]) -> str:
    from fse._container._hash import Hasher

    hasher = Hasher()
    for row in rows:
        data = row.encode("utf-8")
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    return hasher.checksum()


def load_pages(cache: Path, *, download: bool) -> tuple[list[str], np.ndarray]:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    folder = Path(
        snapshot_download(
            REPO,
            repo_type="dataset",
            revision=REVISION,
            cache_dir=cache / "hub",
            allow_patterns=["data/*.parquet"],
            local_files_only=not download,
            max_workers=2,
        )
    )
    pages, scores = [], []
    for number in range(8):
        file = folder / f"data/train-{number:05d}-of-00008.parquet"
        table = pq.read_table(file, columns=["text", "score"]).to_pydict()
        pages.extend(" ".join(text.split()) for text in table["text"])
        scores.extend(table["score"])
    if len(pages) != 467_424 or not all(0 <= score <= 5 for score in scores):
        raise ValueError("unexpected FineWeb annotation count or scores")
    return pages, np.asarray(scores, dtype=np.int8)


def split(pages: list[str], scores: np.ndarray):
    order = np.random.default_rng(0).permutation(len(pages))
    train, evaluation = order[:-EVAL], order[-EVAL:]
    return (
        [pages[i] for i in train],
        scores[train] >= 3,
        [pages[i] for i in evaluation],
        scores[evaluation] >= 3,
    )


def metrics(labels, predicted) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
    }


def record(args, row: dict) -> None:
    with args.results.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps({key: row[key] for key in ("backend", "model", "macro_f1")}), flush=True)


def potion(args, data, provenance: dict, repo: str) -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression

    from fse import Classifier, Encoder

    train, y_train, evaluation, y_eval = data
    encoder = Encoder.load(args.models / artifact_name(repo, "float32"))
    fitted = LogisticRegression(max_iter=1000, random_state=0).fit(
        encoder.encode(train, workers=2), y_train.astype(np.int64)
    )
    stem = repo.rsplit("/", maxsplit=1)[-1]
    joblib.dump(fitted, args.output / f"{stem}-fse-sklearn.joblib")
    reference = None
    for dtype in ("float32", "float16", "int8"):
        enc = Encoder.load(args.models / artifact_name(repo, dtype))
        model = Classifier.from_sklearn(enc, fitted)
        predicted = model.predict(evaluation, workers=2)
        if reference is None:
            reference = predicted
        path = args.output / f"{stem}-{dtype}.safetensors"
        model.save(path)
        record(
            args,
            {
                **provenance,
                **metrics(y_eval, predicted),
                "backend": "fse",
                "model": repo,
                "revision": MODELS[repo],
                "table_dtype": dtype,
                "head": fitted.get_params(),
                "head_training_table_dtype": "float32",
                "encoder_fingerprint": enc.fingerprint,
                "file_bytes": path.stat().st_size,
                "prediction_disagreement_vs_float32": float(np.mean(predicted != reference)),
            },
        )
    native(args, data, provenance, repo)


def native(args, data, provenance: dict, repo: str) -> None:
    import joblib
    from huggingface_hub import snapshot_download
    from model2vec import StaticModel
    from sklearn.linear_model import LogisticRegression

    train, y_train, evaluation, y_eval = data
    folder = snapshot_download(repo, revision=MODELS[repo], local_files_only=True)
    model = StaticModel.from_pretrained(folder)
    fitted = LogisticRegression(max_iter=1000, random_state=0).fit(
        model.encode(train, use_multiprocessing=False), y_train.astype(np.int64)
    )
    predicted = fitted.predict(model.encode(evaluation, use_multiprocessing=False))
    joblib.dump(
        fitted, args.output / f"{repo.rsplit('/', maxsplit=1)[-1]}-model2vec-sklearn.joblib"
    )
    record(
        args,
        {
            **provenance,
            **metrics(y_eval, predicted),
            "backend": "model2vec",
            "model": repo,
            "revision": MODELS[repo],
            "table_dtype": "float32",
            "head": fitted.get_params(),
            "native_multiprocessing": False,
        },
    )


def fasttext(args, data, provenance: dict) -> None:
    import fasttext

    train, y_train, evaluation, y_eval = data
    path = args.output / "owned-intermediate-train.fasttext"
    with path.open("x", encoding="utf-8") as stream:
        for text, label in zip(train, y_train, strict=True):
            stream.write(f"__label__{int(label)} {text}\n")
    try:
        parameters = dict(
            lr=0.1,
            dim=100,
            epoch=5,
            wordNgrams=2,
            loss="softmax",
            bucket=2_000_000,
            thread=2,
            verbose=0,
        )
        model = fasttext.train_supervised(str(path), **parameters)
        predicted = np.array(
            [int(labels[0].removeprefix("__label__")) for labels in model.predict(evaluation)[0]]
        )
        saved = args.output / "fasttext.bin"
        model.save_model(str(saved))
        record(
            args,
            {
                **provenance,
                **metrics(y_eval, predicted),
                "backend": "fasttext",
                "model": "supervised, wordNgrams=2",
                "parameters": parameters,
                "training_determinism": "parallel fastText training is not bit deterministic",
                "file_bytes": saved.stat().st_size,
            },
        )
    finally:
        # This exact intermediate was created exclusively by this invocation.
        print(
            f"removing owned training intermediate {path} ({path.stat().st_size} bytes)", flush=True
        )
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/fse-quality/fineweb")
    parser.add_argument("--models", type=Path, default=Path.home() / ".cache/fse-quality/models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("run with nice -n 10")
    args.output.mkdir()  # Refuse an existing output directory or partially trained artifacts.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = "2"
    from threadpoolctl import threadpool_limits

    with task_reservation(args.coordination), threadpool_limits(limits=2):
        pages, scores = load_pages(args.cache, download=True)
        data = split(pages, scores)
        del pages, scores
        provenance = {
            **identity(),
            "dataset": REPO,
            "dataset_revision": REVISION,
            "train_rows": len(data[0]),
            "evaluation_rows": len(data[2]),
            "train_digest": digest(data[0]),
            "evaluation_digest": digest(data[2]),
            "split": "default_rng(0) permutation, last 46743 held out",
            "label": "score >= 3",
            "preprocessing": "whitespace collapsed",
            "workers": 2,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
        }
        (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for repo in ("minishlab/potion-base-8M", "minishlab/potion-base-32M"):
        with task_reservation(args.coordination), threadpool_limits(limits=2):
            potion(args, data, provenance, repo)
    with task_reservation(args.coordination), threadpool_limits(limits=2):
        fasttext(args, data, provenance)


if __name__ == "__main__":
    main()
