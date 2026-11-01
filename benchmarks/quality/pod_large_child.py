"""One large raw-file to durable ShardSink run, with bounded post-run verification."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_artifacts import file_digest  # isort: skip
from compare_inputs import identity  # isort: skip
from compare_models import MODELS  # isort: skip
from compare_runtime import verify_backend_identity  # isort: skip
from large_write_paths import array_digest, disk, peak_rss, run  # isort: skip


def verify_store(destination, expected):
    import fse
    from fse._container._hash import Hasher

    digest = Hasher()
    with fse.VectorStore.open(destination) as store:
        if len(store) != expected:
            raise ValueError("large pipeline did not persist every input row")
        store.verify()
        for start in range(0, expected, 65536):
            vectors = store.get_vectors(start=start, stop=min(start + 65536, expected))
            if not np.isfinite(vectors).all():
                raise ValueError("nonfinite persisted output")
            digest.update(memoryview(np.ascontiguousarray(vectors)).cast("B"))
    return digest.checksum()


def measure(args, source, path):
    from threadpoolctl import threadpool_limits

    before = disk(args.inputs)
    itemsize = {"float32": 4, "float16": 2, "int8": 1}[args.storage_dtype]
    per_row = 256 * itemsize + (8 if args.storage_dtype == "int8" else 4)
    output_bytes = source["texts"] * per_row if args.mode in ("shard", "npy") else 0
    needed = 2 * output_bytes + 8 * 2**30
    if before["free"] < needed:
        raise RuntimeError(
            f"large write preflight: need {needed} free bytes for two outputs plus reserve"
        )
    with tempfile.TemporaryDirectory(prefix="large-vectors-", dir=args.inputs) as temporary:
        destination = Path(temporary) / "vectors"
        with threadpool_limits(limits=args.workers):
            record, vectors = run(args, destination)
        peak = peak_rss()
        if args.mode == "shard":
            expected_durable = source["texts"] if args.durability == "full" else 0
            if (
                record["published_rows"] != source["texts"]
                or record["durable_rows"] != expected_durable
            ):
                raise ValueError("ShardSink did not publish the required rows/durability")
        occupied = disk(args.inputs)
        digest = verify_output(args, destination, vectors, source["texts"])
        size = (
            sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
            if destination.exists()
            else 0
        )
    if file_digest(path) != source["checksum"]:
        raise ValueError("large input changed during inference")
    return {
        **record,
        "output_digest": digest,
        "peak_rss_bytes": peak,
        "bytes_written": size,
        "file_bytes": size,
        "df_before": before,
        "df_with_output": occupied,
        "df_after_cleanup": disk(args.inputs),
    }


def verify_output(args, destination, vectors, expected):
    if args.kind == "rust":
        vectors = np.load(destination / "vectors.npy", mmap_mode="r")
    if vectors is not None:
        return array_digest(vectors, expected)
    return verify_store(destination, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument(
        "--corpus", choices=["repeated-shuffled-10m", "repeated-shuffled-100m"], required=True
    )
    parser.add_argument("--workers", type=int, choices=[1, 32], required=True)
    parser.add_argument("--kind", choices=["fse", "rust"], default="fse")
    parser.add_argument("--mode", choices=["encode", "memory", "shard", "npy"], default="shard")
    parser.add_argument(
        "--storage-dtype", choices=["float32", "float16", "int8"], default="float32"
    )
    parser.add_argument("--durability", choices=["none", "full"], default="full")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("large timing runs only on RunPod Linux")
    if (args.kind == "rust") != (args.mode == "npy"):
        raise ValueError("Rust runs the NumPy writer; fse runs native encode/memory/shard paths")
    source = json.loads((args.inputs / (args.corpus + ".identity.json")).read_text())
    path = args.inputs / source["file"]
    if file_digest(path) != source["checksum"]:
        raise ValueError("large input differs from its preparation receipt")
    runtime = identity(args.kind)
    record = measure(args, source, path)
    verify_backend_identity(args.kind, runtime)
    print(
        json.dumps(
            {
                **runtime,
                **source,
                **record,
                "kind": args.kind,
                "benchmark": "large-end-to-end",
                "requested_workers": args.workers,
                "rust_parallelism": {
                    "RAYON_NUM_THREADS": str(args.workers),
                    "TOKENIZERS_PARALLELISM": "true" if args.workers > 1 else "false",
                    "observed_thread_count": None,
                }
                if args.kind == "rust"
                else None,
                "fse_workers": args.workers if args.kind == "fse" else None,
                "model": "minishlab/potion-base-8M",
                "revision": MODELS["minishlab/potion-base-8M"],
                "table_dtype": "float32",
                "output_dtype": args.storage_dtype if args.mode == "shard" else "float32",
                "mode": args.mode,
                "corpus": args.corpus,
                "durability": args.durability,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "texts_per_second": source["texts"] / record["p50_s"],
                "tokens_per_second": source["potion_tokens"]["total"] / record["p50_s"],
                "mean_tokens_per_text": source["potion_tokens"]["mean"],
                "rss_scope": "fresh Python inference child or Rust subprocess high-water mark before verification",
                "input_format": "native text path"
                if args.kind == "fse"
                else "Rust BufReader lines",
                "scope": "input read, encode, requested sink/writer and close; model load and digest verification excluded",
                "output_digest_scope": "row-major float32 vector bytes; ShardSink values decoded after stored shard checksum verification",
                "bytes_written_scope": "logical output file bytes, not device write amplification",
                "output_retention": "each output verified, measured, then removed before the next operation",
            }
        )
    )


if __name__ == "__main__":
    main()
