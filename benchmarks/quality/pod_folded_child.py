"""Inference only: folded linear, raw encode plus identical linear matmul, exported MLP."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_artifacts import file_digest  # isort: skip
from compare_inputs import identity  # isort: skip
from compare_runtime import backend_identity, verify_backend_identity  # isort: skip
from compare_models import MODELS  # isort: skip
from compare_snapshots import cached_snapshot  # isort: skip
from folded_classifier_prepare import MLP_DESCRIPTION, MLP_SOURCE, REPO, linear_predict  # isort: skip
from native_classifier import checked_pipeline  # isort: skip
from prepare_fineweb import digest, metrics  # isort: skip


def evaluation(folder, manifest):
    path = folder / manifest["evaluation"]["file"]
    if file_digest(path) != manifest["evaluation"]["checksum"]:
        raise ValueError("evaluation file differs from its preparation record")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    texts, labels = [row["text"] for row in rows], np.asarray([row["label"] for row in rows])
    if (
        len(texts) != manifest["evaluation"]["rows"]
        or digest(texts) != manifest["evaluation"]["input_digest"]
    ):
        raise ValueError("evaluation rows differ from the complete pinned test set")
    return texts, labels


def predictor(kind, folder, workers):
    import fse

    if kind == "mlp":
        pipeline, _ = checked_pipeline(folder / "model2vec-mlp")
        return lambda rows: pipeline.predict(rows, use_multiprocessing=workers > 1)
    if kind == "model2vec-linear":
        import joblib
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(cached_snapshot(REPO, MODELS[REPO]), normalize=False)
        head = joblib.load(folder / "same-sklearn-head.joblib")
        with np.load(folder / "linear-head.npz") as recorded:
            if not np.array_equal(head.coef_, recorded["weight"]) or not np.array_equal(
                head.intercept_, recorded["bias"]
            ):
                raise ValueError("sklearn comparator does not carry the same float32 linear head")
        return lambda rows: head.predict(
            model.encode(rows, normalize=False, use_multiprocessing=workers > 1)
        )
    if kind == "folded":
        model = fse.Classifier.load(folder / "folded-linear.safetensors", mmap=True)
        if not model.standalone or model.normalized:
            raise ValueError("folded arm must load the standalone raw-mean logit table")
        return lambda rows: model.predict(rows, workers=workers)
    encoder = fse.Encoder.load(folder / "raw-encoder.safetensors", mmap=True)
    if encoder.steps:
        raise ValueError("encode-plus-head arm must use raw means with no normalization")
    with np.load(folder / "linear-head.npz") as data:
        weight, bias, classes = data["weight"], data["bias"], data["classes"]
    if weight.dtype != np.float32 or bias.dtype != np.float32:
        raise ValueError("explicit matmul must use the exact float32 fse head")
    return lambda rows: linear_predict(encoder, weight, bias, classes, rows, workers)


def measure(predict, rows):
    start = time.perf_counter()
    expected = predict(rows)
    first = time.perf_counter() - start
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        observed = predict(rows)
        durations.append(time.perf_counter() - start)
        if not np.array_equal(observed, expected):
            raise ValueError("predictions changed across inference passes")
    return expected, first, durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=["folded", "encode-matmul", "model2vec-linear", "mlp"], required=True
    )
    parser.add_argument("--workers", type=int, choices=[1, 32], required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("classifier timing is restricted to the Linux pod")
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if args.workers > 1 else "false"
    os.environ["RAYON_NUM_THREADS"] = str(args.workers)
    from threadpoolctl import threadpool_info, threadpool_limits

    from fse._container._hash import checksum

    manifest = json.loads((args.folder / "manifest.json").read_text())
    rows, labels = evaluation(args.folder, manifest)
    for name, expected in manifest["files"].items():
        if file_digest(args.folder / name) != expected["checksum"]:
            raise ValueError("fitted linear artifact changed before inference")
    backend = "model2vec" if args.kind in ("mlp", "model2vec-linear") else "fse"
    runtime = identity(backend)
    head_runtime = backend_identity("tfidf") if args.kind == "model2vec-linear" else None
    with threadpool_limits(limits=args.workers):
        start = time.perf_counter()
        predict = predictor(args.kind, args.folder, args.workers)
        load_s = time.perf_counter() - start
        predicted, first, seconds = measure(predict, rows)
        pools = threadpool_info()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    verify_backend_identity(backend, runtime)
    if head_runtime is not None:
        verify_backend_identity("tfidf", head_runtime)
    for name, expected in manifest["files"].items():
        if file_digest(args.folder / name) != expected["checksum"]:
            raise ValueError("fitted linear artifact changed during inference")
    sizes = {name: row["bytes"] for name, row in manifest["files"].items()}
    row = {
        **runtime,
        "kind": args.kind,
        "benchmark": "classifier-folding",
        "suite": manifest["suite"],
        "dataset": manifest["dataset_name"],
        "model": manifest["model"],
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "revision": manifest["revision"],
        "texts": len(rows),
        "input_digest": digest(rows),
        "output_digest": checksum(np.ascontiguousarray(predicted)),
        "requested_workers": args.workers,
        "table_dtype": "float32",
        "output_dtype": "labels",
        "load_s": load_s,
        "first_predict_s": first,
        "warm_seconds": seconds,
        "p50_s": float(np.median(seconds)),
        "p95_s": float(np.percentile(seconds, 95)),
        "texts_per_second": len(rows) / float(np.median(seconds)),
        "peak_rss_bytes": peak,
        "rss_scope": "fresh inference process before accuracy-library import",
        "quality": metrics(labels, predicted),
        "threadpools": pools,
        "head": MLP_DESCRIPTION
        if args.kind == "mlp"
        else "frozen-embedding linear head, trained on raw unnormalized means",
        "mlp_source": MLP_SOURCE,
        "linear_head_runtime": head_runtime,
        "normalization": "MLP training and exported encoder normalize"
        if args.kind == "mlp"
        else "raw means, normalization disabled",
        "folded_values_per_token": manifest["linear_head"]["outputs"]
        if args.kind == "folded"
        else None,
        "folded_model_file_bytes": sizes["folded-linear.safetensors"],
        "encoder_plus_head_file_bytes": sizes["raw-encoder.safetensors"] + sizes["linear-head.npz"],
        "training_record": manifest,
    }
    print(json.dumps(row))


if __name__ == "__main__":
    main()
