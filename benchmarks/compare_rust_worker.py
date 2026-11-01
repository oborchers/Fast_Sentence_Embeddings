"""Run the pinned native worker without counting JSON or vector IO as encoding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from bench_model2vec import _snapshot
from compare_rust_build import verified_binary


def measure(args, sentences: list[str]) -> tuple[dict, np.ndarray]:
    import resource

    binary, build = verified_binary()
    scratch = Path(os.environ.get("FSE_QUALITY_CACHE", Path.home() / ".cache/fse-quality")) / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        folder = Path(directory)
        corpus = folder / "corpus.json"
        corpus.write_text(json.dumps(sentences), encoding="utf-8")
        output = folder / "vectors.f32"
        environment = {
            **os.environ,
            "RAYON_NUM_THREADS": str(args.threads),
            "TOKENIZERS_PARALLELISM": "false" if args.threads == 1 else "true",
        }
        done = subprocess.run(
            [str(binary), str(_snapshot(args.model)), str(corpus), str(args.passes), str(output)],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        record = json.loads(done.stdout)
        vectors = np.fromfile(output, dtype="<f4").reshape(record["shape"])
    record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    record["rust_worker_build"] = build
    record["rust_source_revision"] = "19f58bef6ce145f2bc08de7016498f28d521feea"
    record["rss_scope"] = "native Rust process"
    record["table_dtype"] = "float32"
    record["output_dtype"] = str(vectors.dtype)
    record["file_bytes"] = (_snapshot(args.model) / "model.safetensors").stat().st_size
    record["load_source"] = "pinned Hub snapshot"
    record["rustc"] = subprocess.check_output(["rustc", "--version"], text=True).strip()
    return record, vectors
