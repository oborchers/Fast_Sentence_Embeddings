"""Explicit native memory/store paths and the locked Rust NumPy writer."""

import os
import resource
import shutil
import subprocess
import time

import numpy as np
from compare_artifacts import checked_encoder
from compare_models import artifact_path
from pod_pipeline import MODEL, rust_pipeline


def disk(path):
    usage = shutil.disk_usage(path)
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "df_h": subprocess.check_output(["/usr/bin/df", "-h", str(path)], text=True),
    }


def fse_run(args, destination):
    import fse

    encoder = fse.Encoder.load(artifact_path(args.models, MODEL, "float32"), mmap=True)
    provenance = checked_encoder(encoder, args.models, MODEL, "float32")
    path = args.inputs / (args.corpus + ".txt")
    vectors = None
    read_start = None
    start = time.perf_counter()
    if args.mode == "encode":
        vectors = encoder.encode(path, workers=args.workers)
    elif args.mode == "memory":
        sink = fse.MemorySink(encoder.output_dim)
        encoder.encode(path, workers=args.workers, sink=sink)
        vectors = sink.result()
    else:
        sink = fse.ShardSink(
            destination, encoder, dtype=args.storage_dtype, durability=args.durability
        )
        read_start = time.perf_counter()
        with sink:
            encoder.encode(path, workers=args.workers, sink=sink)
    end = time.perf_counter()
    elapsed = end - start
    read_to_close = end - read_start if read_start is not None else elapsed
    measured = read_to_close if args.corpus == "repeated-shuffled-100m" else elapsed
    counts = (
        {"published_rows": sink.published_rows, "durable_rows": sink.durable_rows}
        if args.mode == "shard"
        else {}
    )
    return {
        "p50_s": measured,
        "including_sink_open_s": elapsed,
        "read_to_close_s": read_to_close,
        "warm_seconds": [measured],
        **counts,
        "artifact_provenance": provenance,
        "encoder_fingerprint": encoder.fingerprint,
        "time_until_durable_s": measured
        if args.mode == "shard" and args.durability == "full"
        else None,
        "writer": "fse " + args.mode,
        "output_storage_dtype": args.storage_dtype if args.mode == "shard" else "float32",
    }, vectors


def run(args, destination):
    os.environ["RAYON_NUM_THREADS"] = str(args.workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if args.workers > 1 else "false"
    if args.kind == "rust":
        return rust_pipeline(args, destination), None
    return fse_run(args, destination)


def peak_rss():
    return (
        max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        * 1024
    )


def array_digest(vectors, expected):
    from fse._container._hash import Hasher

    if vectors.ndim != 2 or len(vectors) != expected:
        raise ValueError("output does not contain every vector")
    digest = Hasher()
    for start in range(0, expected, 65536):
        block = vectors[start : start + 65536]
        if not np.isfinite(block).all():
            raise ValueError("nonfinite vector output")
        digest.update(memoryview(np.ascontiguousarray(block)).cast("B"))
    return digest.checksum()
