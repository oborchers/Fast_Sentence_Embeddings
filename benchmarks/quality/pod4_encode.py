"""Large identical-input encoding operations, never run on the local Mac."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_compare import python_measure  # isort: skip
from artifact_manifest import header
from compare_artifacts import file_digest
from compare_models import artifact_path

from pod4_memory import MemoryAdmissionError, encoding  # isort: skip
from compare_inputs import identity  # isort: skip
from compare_models import MODELS  # isort: skip
from compare_parallelism import settings  # isort: skip
from compare_runtime import verify_backend_identity  # isort: skip


def command(options, repo, dtype, workers, kind):
    return [
        sys.executable,
        __file__,
        "--kind",
        kind,
        "--threads",
        str(workers),
        "--model",
        repo,
        "--table-dtype",
        dtype,
        "--models",
        str(options.models),
        "--input",
        str(options.input_file),
        "--minimum-seconds",
        "2",
    ]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=["fse", "model2vec", "rust", "fastembed", "st"], required=True
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--table-dtype", required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--minimum-seconds", type=float, default=2)
    parser.add_argument("--vectors-out", type=Path)
    parser.add_argument("--passes", type=int, default=1)
    return parser.parse_args()


def main():
    from threadpoolctl import threadpool_info, threadpool_limits

    args = arguments()
    if sys.platform != "linux":
        raise RuntimeError("Pod 4 encoding runs only on Linux")
    allowed = sorted(os.sched_getaffinity(0))
    if args.threads < 1 or len(allowed) < args.threads:
        raise ValueError("requested CPU budget exceeds the pod affinity")
    os.sched_setaffinity(0, allowed[: args.threads])
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if args.threads > 1 else "false"
    source = json.loads(args.input.with_suffix(".identity.json").read_text())
    if source["checksum"] != file_digest(args.input):
        raise ValueError("shared input differs from its recorded file digest")
    artifact = artifact_path(args.models, args.model, args.table_dtype)
    width = int(header(artifact)["fse.output_dim"])
    admission = encoding(source["texts"], width, args.input.stat().st_size, artifact.stat().st_size)
    before = identity(args.kind)
    if args.kind == "rust":
        from compare_rust_worker import measure

        options = SimpleNamespace(
            model=args.model, threads=args.threads, passes=args.passes, input_file=args.input
        )
        record, vectors = measure(options, [])
    else:
        with args.input.open(encoding="utf-8") as stream:
            sentences = [line.rstrip("\n") for line in stream]
        if len(sentences) != source["texts"]:
            raise ValueError("input row count differs from its receipt")
        options = SimpleNamespace(
            kind=args.kind,
            threads=args.threads,
            model=args.model,
            passes=args.passes,
            table_dtype=args.table_dtype,
            fse_models=args.models,
            warmup_texts=None,
        )
        with threadpool_limits(limits=args.threads):
            record, vectors = python_measure(options, sentences)
            record["threadpools"] = threadpool_info()
    if len(vectors) != source["texts"]:
        raise ValueError("encoded row count differs from the complete input")
    verify_backend_identity(args.kind, before)
    if args.vectors_out:
        np.save(args.vectors_out, vectors)
    record["memory_admission"] = admission
    print(json.dumps(finish(args, source, before, vectors, record)))


def finish(args, source, before, vectors, record):
    import time

    from fse._container._hash import checksum

    record.update(
        {
            **before,
            **{
                key: value
                for key, value in source.items()
                if key not in ("file_bytes", "file", "checksum")
            },
            "input_file_bytes": source["file_bytes"],
            "input_checksum": source["checksum"],
            "effective_affinity": sorted(os.sched_getaffinity(0)),
            **settings(args.kind, args.threads, source["texts"]),
            "kind": args.kind,
            "model": args.model,
            "revision": MODELS[args.model],
            "corpus": "wikitext-103-train-passages",
            "shape": list(vectors.shape),
            "output_digest": checksum(np.ascontiguousarray(vectors)),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "p50_s": float(np.median(record["warm_seconds"])),
            "minimum_call_seconds": args.minimum_seconds if args.kind == "fse" else 0,
            "timed_passes": args.passes,
        }
    )
    return record


if __name__ == "__main__":
    try:
        main()
    except MemoryAdmissionError as error:
        print(
            json.dumps({"event": "memory_admission_rejected", "budget": error.receipt}), flush=True
        )
        raise SystemExit(78) from error
