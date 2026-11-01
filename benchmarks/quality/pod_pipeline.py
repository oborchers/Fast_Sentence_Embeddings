"""Raw file to persisted vectors. Setup is separate, input reads and close are timed."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_backends import make_contender  # isort: skip
from compare_artifacts import checked_encoder  # isort: skip
from compare_inputs import corpus_identity, identity, texts  # isort: skip
from compare_models import artifact_path  # isort: skip
from compare_runtime import verify_backend_identity  # isort: skip

MODEL = "minishlab/potion-base-8M"


def canonical_inputs(corpus, folder):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    original = texts(corpus)
    rows = [row.replace("\r", " ").replace("\n", " ") for row in original]
    source = corpus_identity(corpus, rows)
    source["original_input_digest"] = corpus_identity(corpus, original)["input_digest"]
    source["preprocessing"] = (
        "CR and LF replaced individually with spaces, no stripping, no dropped rows"
    )
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{corpus}.txt").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(row + "\n")
    table = pa.table({"text": rows})
    with ipc.new_file(folder / f"{corpus}.arrow", table.schema) as writer:
        writer.write_table(table)
    (folder / f"{corpus}.identity.json").write_text(json.dumps({**source, "texts": len(rows)}))


def fse_pipeline(args, destination):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    import fse

    encoder = fse.Encoder.load(artifact_path(args.models, MODEL, "float32"))
    provenance = checked_encoder(encoder, args.models, MODEL, "float32")
    start = time.perf_counter()
    if args.input_format == "arrow":
        with pa.memory_map(str(args.inputs / f"{args.corpus}.arrow"), "r") as source:
            column = ipc.open_file(source).read_all().column("text")
            with fse.ShardSink(destination, encoder, durability=args.durability) as sink:
                encoder.encode(column, workers=args.workers, sink=sink)
    else:
        with fse.ShardSink(destination, encoder, durability=args.durability) as sink:
            encoder.encode(args.inputs / f"{args.corpus}.txt", workers=args.workers, sink=sink)
    elapsed = time.perf_counter() - start
    return {
        "p50_s": elapsed,
        "warm_seconds": [elapsed],
        "artifact_provenance": provenance,
        "time_until_durable_s": elapsed if args.durability == "full" else None,
        "writer": "fse.ShardSink, close included",
        "encoder_fingerprint": encoder.fingerprint,
        "published_rows": sink.published_rows,
        "durable_rows": sink.durable_rows,
    }


def numpy_pipeline(args, destination):
    encode = make_contender(args.kind, args.workers, MODEL)
    start = time.perf_counter()
    with (args.inputs / f"{args.corpus}.txt").open(encoding="utf-8", newline="") as stream:
        rows = [line[:-1] if line.endswith("\n") else line for line in stream]
    vectors = encode(rows)
    destination.mkdir()
    with (destination / "vectors.npy").open("xb") as stream:
        np.save(stream, vectors)
        stream.flush()
        if args.durability == "full":
            os.fsync(stream.fileno())
    if args.durability == "full":
        for directory in (destination, destination.parent):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    elapsed = time.perf_counter() - start
    return {
        "p50_s": elapsed,
        "warm_seconds": [elapsed],
        "time_until_durable_s": elapsed if args.durability == "full" else None,
        "writer": "Python line reading + library encode + np.save, flush, optional file and directory fsync",
    }


def rust_pipeline(args, destination):
    from compare_backends import pinned_snapshot
    from compare_rust_build import verified_binary

    binary, _ = verified_binary()
    command = [
        str(binary),
        "--pipeline",
        str(pinned_snapshot(MODEL)),
        str(args.inputs / f"{args.corpus}.txt"),
        str(destination / "vectors.npy"),
        args.durability,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["fse", "model2vec", "rust", "fastembed"], required=True)
    parser.add_argument("--corpus", choices=["wikitext", "massive"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 32], required=True)
    parser.add_argument("--input-format", choices=["text", "arrow"], default="text")
    parser.add_argument("--durability", choices=["full", "none"], required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("pipeline timing is restricted to the Linux pod")
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if args.workers > 1 else "false"
    os.environ["RAYON_NUM_THREADS"] = str(args.workers)
    from threadpoolctl import threadpool_limits

    provenance = identity(args.kind)
    source = json.loads((args.inputs / f"{args.corpus}.identity.json").read_text())
    with (
        tempfile.TemporaryDirectory(prefix="pipeline-", dir=args.inputs) as temporary,
        threadpool_limits(limits=args.workers),
    ):
        destination = Path(temporary) / "vectors"
        if args.kind == "fse":
            record = fse_pipeline(args, destination)
        elif args.kind == "rust":
            record = rust_pipeline(args, destination)
        else:
            record = numpy_pipeline(args, destination)
        peak = max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        record["output_digest"] = persisted_digest(destination, args.kind, source["texts"])
        record["file_bytes"] = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    verify_backend_identity(args.kind, provenance)
    print(
        json.dumps(
            {
                **provenance,
                **source,
                **record,
                "benchmark": "end-to-end",
                "peak_rss_bytes": peak * 1024,
                "rss_scope": "maximum fresh Python parent or Rust child process high-water mark",
                "texts_per_second": source["texts"] / record["p50_s"],
                "model": MODEL,
                "kind": args.kind,
                "workers": args.workers,
                "input_format": args.input_format,
                "durability": args.durability,
                "table_dtype": "float32",
            }
        )
    )


def persisted_digest(destination, kind, expected_rows):
    import fse
    from fse._container._hash import checksum

    if kind == "fse":
        with fse.VectorStore.open(destination) as store:
            store.verify()
            vectors = store.get_vectors()
    else:
        vectors = np.load(destination / "vectors.npy", mmap_mode="r")
    if len(vectors) != expected_rows or not np.isfinite(vectors).all():
        raise ValueError("persisted vectors do not contain every finite output row")
    return checksum(np.ascontiguousarray(vectors))


if __name__ == "__main__":
    main()
