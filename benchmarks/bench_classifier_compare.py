"""Locked rotating classifier pairs after offline preparation has completed.

SetFit reconstruction queries all fifteen full test sets. FineWeb queries every
held-out page. Timers include tokenization, embeddings and label prediction.
The reported aggregate rate divides total test rows by the sum of prediction
seconds, not an average of per-dataset rates. Per-dataset times are retained.
Model loading and first full prediction are recorded separately. No fitting is
inside prediction timing. Run with the isolated comparison environment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np
from bench_compare import append, reservation
from compare_artifacts import (
    check_head_encoder,
    checked_encoder,
    file_digest,
    native_encoder_identity,
    validate_preparation,
)
from compare_inputs import identity
from compare_models import MODELS, artifact_name
from compare_parallelism import settings
from compare_runtime import verify_backend_identity
from compare_search import timed
from compare_wait import ready_operation

sys.path.insert(0, str(Path(__file__).resolve().parent / "quality"))
from prepare_fineweb import digest, load_pages, split, REVISION as FINEWEB_REVISION  # isort: skip
from prepare_setfit import SOURCES, dataset  # isort: skip


def evaluation_sets(args):
    if args.suite == "fineweb":
        pages, scores = load_pages(args.cache, download=False)
        _, _, evaluation, _ = split(pages, scores)
        return [("fineweb", evaluation)]
    specifications = json.loads(SOURCES.read_text())["datasets"]
    return [
        (spec["dataset"].removeprefix("SetFit/"), dataset(spec, args.cache, download=False)[2])
        for spec in specifications
    ]


def predictor(args, name):
    import joblib

    from fse import Classifier, Encoder

    folder = args.classifiers if args.suite == "fineweb" else args.classifiers / name
    stem = args.model.rsplit("/", maxsplit=1)[-1]
    if args.kind == "fse":
        if args.suite == "fineweb":
            model = Classifier.load(folder / f"{stem}-{args.table_dtype}.safetensors")
        else:
            encoder = Encoder.load(args.models / artifact_name(args.model, args.table_dtype))
            head = joblib.load(folder / f"{stem}-fse-sklearn.joblib")
            model = Classifier.from_sklearn(encoder, head)
        source = checked_encoder(model.encoder, args.models, args.model, args.table_dtype)
        head_name = (
            f"{stem}-{args.table_dtype}.safetensors"
            if args.suite == "fineweb"
            else f"{stem}-fse-sklearn.joblib"
        )
        check_head_encoder(json.loads((folder / "manifest.json").read_text()), head_name, source)
        if model.encoder.word_vectors.dtype != args.table_dtype:
            raise ValueError("classifier table dtype differs from the requested dtype")
        return lambda rows: model.predict(rows, workers=args.workers), fse_files(args, folder, stem)
    if args.kind == "model2vec":
        from huggingface_hub import snapshot_download
        from model2vec import StaticModel

        folder_model = snapshot_download(
            args.model, revision=MODELS[args.model], local_files_only=True
        )
        model = StaticModel.from_pretrained(folder_model)
        source = native_encoder_identity(args.model, Path(folder_model))
        check_head_encoder(
            json.loads((folder / "manifest.json").read_text()),
            f"{stem}-model2vec-sklearn.joblib",
            source,
        )
        head = joblib.load(folder / f"{stem}-model2vec-sklearn.joblib")
        return (
            lambda rows: head.predict(model.encode(rows, use_multiprocessing=False)),
            [
                *Path(folder_model).rglob("*.safetensors"),
                *Path(folder_model).rglob("*.json"),
                folder / f"{stem}-model2vec-sklearn.joblib",
            ],
        )
    if args.kind == "tfidf":
        pipeline = joblib.load(folder / "tfidf-sklearn.joblib")
        return pipeline.predict, [folder / "tfidf-sklearn.joblib"]
    import fasttext

    model = fasttext.load_model(str(folder / "fasttext.bin"))

    def predict(rows):
        return np.array(
            [int(labels[0].removeprefix("__label__")) for labels in model.predict(rows)[0]]
        )

    return predict, [folder / "fasttext.bin"]


def fse_files(args, folder, stem):
    if args.suite == "fineweb":
        return [folder / f"{stem}-{args.table_dtype}.safetensors"]
    return [
        args.models / artifact_name(args.model, args.table_dtype),
        folder / f"{stem}-fse-sklearn.joblib",
    ]


def parallelism(args):
    result = settings(args.kind, args.workers, 0)
    if args.kind == "model2vec":
        result.update(
            tokenizer_parallelism="on" if args.workers > 1 else "off",
            parallel_batch_pool="disabled",
        )
    if args.kind == "fasttext":
        result["inference_parallelism"] = "single caller, native batch predict API"
    return result


def preparation_info(args, name, rows):
    if args.suite == "fineweb":
        folder, revision = args.classifiers, FINEWEB_REVISION
    else:
        specification = next(
            item
            for item in json.loads(SOURCES.read_text())["datasets"]
            if item["dataset"] == "SetFit/" + name
        )
        folder, revision = args.classifiers / name, specification["revision"]
    return validate_preparation(folder, revision, digest(rows))


def artifact_identities(paths, preparation):
    result = {}
    for path in paths:
        observed = {"file_bytes": path.stat().st_size, "checksum": file_digest(path)}
        expected = preparation.get("artifacts", {}).get(path.name)
        if expected is None and (
            path.suffix in (".joblib", ".bin") or path.name in preparation.get("head_encoders", {})
        ):
            raise ValueError(f"{path.name}: fitted artifact missing from preparation manifest")
        if expected is not None and observed != expected:
            raise ValueError(f"{path.name}: fitted artifact differs from preparation manifest")
        result[str(path)] = observed
    return result


def measure(args, sets):
    from fse._container._hash import checksum

    records = []
    for name, rows in sets:
        preparation = preparation_info(args, name, rows)
        seconds, (predict, paths) = timed(lambda n=name: predictor(args, n))
        identities = artifact_identities(paths, preparation)
        first_s, expected = timed(lambda p=predict, r=rows: p(r))
        output_digest = checksum(np.ascontiguousarray(expected))
        times = []
        for _ in range(args.passes):
            duration, predicted = timed(lambda p=predict, r=rows: p(r))
            if not np.array_equal(predicted, expected):
                raise ValueError("classifier predictions changed between repeated calls")
            times.append(duration)
        records.append(
            {
                "dataset": name,
                "preparation_manifest": preparation,
                "artifact_identities": identities,
                "texts": len(rows),
                "input_digest": digest(rows),
                "model_setup_s": seconds,
                "first_predict_s": first_s,
                "warm_seconds": times,
                "prediction_digest": output_digest,
                "artifacts": {str(path): path.stat().st_size for path in paths},
            }
        )
        del predict, expected, predicted
    return records


def child(args):
    import resource

    from threadpoolctl import threadpool_info, threadpool_limits

    os.environ["TOKENIZERS_PARALLELISM"] = "false" if args.workers == 1 else "true"
    os.environ["RAYON_NUM_THREADS"] = str(args.workers)
    sets = evaluation_sets(args)
    manifest = (args.classifiers / "manifest.json") if args.suite == "fineweb" else SOURCES
    provenance = identity(args.kind)
    with threadpool_limits(limits=args.workers):
        records = measure(args, sets)
        verify_backend_identity(args.kind, provenance)
        passes = np.sum([row["warm_seconds"] for row in records], axis=0)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {
            **provenance,
            "suite": args.suite,
            "model": args.model,
            "revision": MODELS[args.model],
            "kind": args.kind,
            "corpus": args.suite,
            "texts": sum(len(rows) for _, rows in sets),
            "input_digest": digest([row["input_digest"] for row in records]),
            "table_dtype": (
                args.table_dtype
                if args.kind == "fse"
                else (None if args.kind == "tfidf" else "float32")
            ),
            "output_dtype": "labels",
            "threads": args.workers,
            **parallelism(args),
            "native_multiprocessing": False,
            "timed_passes": args.passes,
            "load_s": sum(row["model_setup_s"] for row in records),
            "first_predict_s": sum(row["first_predict_s"] for row in records),
            "p50_s": float(np.median(passes)),
            "p95_s": float(np.percentile(passes, 95)),
            "warm_seconds": passes.tolist(),
            "datasets": records,
            "file_bytes": sum(
                {path: size for row in records for path, size in row["artifacts"].items()}.values()
            ),
            "file_bytes_scope": "required weights, tokenizer/config JSON and fitted heads, unique paths",
            "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "rss_scope": "fresh process, complete test inputs and one classifier at a time",
            "threadpools": threadpool_info(),
            "preparation_manifest": json.loads(manifest.read_text()),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    print(json.dumps(record))


def operation(args, label):
    if subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "src",
            ":(glob)benchmarks/**/*.py",
            "pyproject.toml",
            "setup.py",
            "uv.lock",
        ],
        check=False,
    ).returncode:
        raise RuntimeError("timing deferred: tracked changes are not committed")
    kind = "fse" if label.startswith("fse") else args.against
    command = [
        sys.executable,
        __file__,
        "child",
        "--kind",
        kind,
        "--suite",
        args.suite,
        "--workers",
        str(args.workers),
        "--model",
        args.model,
        "--models",
        str(args.models),
        "--classifiers",
        str(args.classifiers),
        "--cache",
        str(args.cache),
        "--table-dtype",
        args.table_dtype,
        "--passes",
        str(args.passes),
    ]
    with reservation(args.lock):
        before = os.getloadavg()[0]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        after = os.getloadavg()[0]
        if done.returncode:
            raise RuntimeError(f"{label} failed: {done.stderr[-3000:]}")
        return {
            **json.loads(done.stdout.strip().splitlines()[-1]),
            "label": label,
            "load_before": before,
            "load_after": after,
            "valid": before < 5 and after < 5,
        }


def run(args):
    series, valid = uuid.uuid4().hex, 0
    deadline = time.monotonic() + args.budget_s
    order = ["fse-a", "fse-b", args.against]
    for attempt in range(6):
        records = []
        try:
            for label in order:
                row = {
                    **ready_operation(args.lock, deadline, partial(operation, args, label)),
                    "event": "operation",
                    "series_id": series,
                    "round": attempt,
                }
                append(args.results, row)
                records.append(row)
        except (RuntimeError, FileExistsError) as error:
            append(
                args.results,
                {
                    "event": "round_complete",
                    "series_id": series,
                    "round": attempt,
                    "complete": False,
                    "valid": False,
                    "reason": str(error),
                },
            )
            raise
        if len({row["input_digest"] for row in records}) != 1:
            raise ValueError("classifiers did not receive the same complete inputs")
        good = all(row["valid"] for row in records)
        append(
            args.results,
            {
                "event": "round_complete",
                "series_id": series,
                "round": attempt,
                "complete": True,
                "valid": good,
            },
        )
        valid += good
        if valid >= 3:
            append(
                args.results,
                {
                    "event": "series_complete",
                    "series_id": series,
                    "valid_rounds": valid,
                    "required_rounds": 3,
                },
            )
            return
        order = order[1:] + order[:1]
    raise RuntimeError("fewer than three valid rounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "child"])
    parser.add_argument("--kind", choices=["fse", "model2vec", "tfidf", "fasttext"], default="fse")
    parser.add_argument(
        "--against", choices=["model2vec", "tfidf", "fasttext"], default="model2vec"
    )
    parser.add_argument("--suite", choices=["setfit", "fineweb"], required=True)
    parser.add_argument("--classifiers", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["minishlab/potion-base-8M", "minishlab/potion-base-32M"],
        default="minishlab/potion-base-8M",
    )
    parser.add_argument("--models", type=Path, default=Path.home() / ".cache/fse-quality/models")
    parser.add_argument("--workers", type=int, choices=[1, 10], default=1)
    parser.add_argument("--table-dtype", choices=["float32", "float16", "int8"], default="float32")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--budget-s", type=float, default=600)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.passes < 3:
        parser.error("passes must be at least three")
    if args.suite == "fineweb" and (args.kind == "tfidf" or args.against == "tfidf"):
        parser.error("TF-IDF is prepared for the SetFit suite only")
    if args.suite == "setfit" and (args.kind == "fasttext" or args.against == "fasttext"):
        parser.error("fastText is prepared for FineWeb only")
    if args.command == "run" and (args.lock is None or args.results is None):
        parser.error("run requires --lock and --results")
    (run if args.command == "run" else child)(args)


if __name__ == "__main__":
    main()
