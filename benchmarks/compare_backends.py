"""Native Python counterparts with explicit CPU parallelism and local pinned models."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from bench_model2vec import _snapshot, contender
from compare_artifacts import checked_encoder
from compare_models import MODELS, artifact_name


def pinned_snapshot(repo: str) -> Path:
    from compare_snapshots import cached_snapshot

    return cached_snapshot(repo, MODELS[repo])


def transformer_contender(threads: int, repo: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if threads == 1 else "true"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    model = SentenceTransformer(str(pinned_snapshot(repo)), device="cpu")
    batch_size = 32 if repo.endswith("all-MiniLM-L6-v2") else 1024

    def encode(batch):
        return model.encode(batch, batch_size=batch_size)

    encode.batch_size = batch_size
    encode.torch_num_threads = torch.get_num_threads()
    return encode


def weights_bytes(folder: Path) -> int:
    files = list(folder.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {folder}")
    return sum(path.stat().st_size for path in files)


def make_contender(kind: str, threads: int, repo: str, *, table_dtype="float32", folder=None):
    if kind == "fse":
        return fse_contender(threads, repo, table_dtype, folder)
    if kind != "fastembed":
        encode = (
            transformer_contender(threads, repo) if kind == "st" else contender(kind, threads, repo)
        )
        encode.table_dtype = "float32"
        encode.file_bytes = weights_bytes(pinned_snapshot(repo))
        encode.load_source = "pinned Hub snapshot"
        return encode
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if threads == 1 else "true"
    from fastembed import TextEmbedding

    # The static model does not accept an ONNX threads argument. It uses the
    # tokenizer pool and numpy BLAS. Both are bounded by the caller.
    model = TextEmbedding(
        model_name=repo, specific_model_path=str(_snapshot(repo)), local_files_only=True
    )

    def encode(batch):
        return np.stack(list(model.embed(batch, parallel=None)))

    encode.table_dtype = "float32"
    encode.file_bytes = weights_bytes(_snapshot(repo))
    encode.load_source = "pinned Hub snapshot"
    return encode


def fse_contender(threads: int, repo: str, dtype: str, folder: Path | None):
    from fse import Encoder

    if folder is not None:
        path = folder / artifact_name(repo, dtype)
        encoder = Encoder.load(path)
        source = "finished fse artifact"
    elif dtype == "float32":
        encoder = Encoder.from_pretrained(repo, revision=MODELS[repo], local_files_only=True)
        path = _snapshot(repo) / "model.safetensors"
        source = "pinned Hub conversion"
    else:
        raise ValueError("quantized table timing requires --fse-models with finished artifacts")
    if encoder.word_vectors.dtype != dtype:
        raise ValueError(f"loaded table is {encoder.word_vectors.dtype}, requested {dtype}")

    provenance = checked_encoder(encoder, folder, repo, dtype) if folder is not None else None

    def encode(batch):
        return encoder.encode(batch, workers=threads)

    encode.table_dtype = encoder.word_vectors.dtype
    encode.file_bytes = path.stat().st_size
    encode.load_source = source
    encode.encoder_fingerprint = encoder.fingerprint
    encode.artifact_provenance = provenance
    return encode
