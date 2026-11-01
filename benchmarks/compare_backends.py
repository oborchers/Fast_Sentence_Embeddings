"""Native Python counterparts with explicit CPU parallelism and local pinned models."""

from __future__ import annotations

import os

import numpy as np
from bench_model2vec import _snapshot, contender


def make_contender(kind: str, threads: int, repo: str):
    if kind != "fastembed":
        return contender(kind, threads, repo)
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if threads == 1 else "true"
    from fastembed import TextEmbedding

    # The static model does not accept an ONNX threads argument. It uses the
    # tokenizer pool and numpy BLAS. Both are bounded by the caller.
    model = TextEmbedding(
        model_name=repo, specific_model_path=str(_snapshot(repo)), local_files_only=True
    )
    return lambda batch: np.stack(list(model.embed(batch, parallel=None)))
