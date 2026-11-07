"""Private identity of an evaluated model: training relation, execution path and files.

These records explain a score; they never change it. They are read once per loaded model,
except the training relation, which is per task."""

from __future__ import annotations

import sys

from _measurement import sha256
from _measurement_config import MODEL_PINS

UNKNOWN, OVERLAP, ZERO_SHOT = "unknown", "training_overlap", "zero_shot"


def training_relation(model, task):
    """Whether the pinned MTEB ModelMeta of ``model`` names training data related to
    ``task``: ``zero_shot``, ``training_overlap`` or ``unknown`` (the model is not in the
    registry or declares no training data). MTEB resolves related tasks and the training
    data of the model it was adapted from."""
    import mteb

    try:
        meta = mteb.get_model_meta(model, fetch_from_hf=False)
    except (KeyError, ValueError):
        return {"state": UNKNOWN, "reason": "model not in the pinned MTEB registry"}
    if meta.name != model:
        return {"state": UNKNOWN, "reason": "registry resolved another model name"}
    datasets = meta.get_training_datasets()
    record = {"registry_revision": meta.revision, "pinned_revision": MODEL_PINS[model]}
    if datasets is None:
        return {**record, "state": UNKNOWN, "reason": "no training data declared"}
    overlap = sorted(datasets & {task})
    return {**record, "state": OVERLAP if overlap else ZERO_SHOT, "overlap": overlap}


def execution(cell, native):
    """What actually runs: the output dimension, the dtypes and the threads of the backend."""
    import numpy as np

    vector = np.asarray(encode_probe(cell, native))
    record = {
        "backend": cell["backend"],
        "output_dim": int(vector.shape[1]),
        "output_dtype": str(vector.dtype),
        "requested_workers": cell["workers"],
        "platform": sys.platform,
    }
    if cell["backend"] == "st":
        import torch

        parameters = list(native.parameters())
        record.update(
            sentence_embedding_dimension=native.get_sentence_embedding_dimension(),
            weight_dtype=str(parameters[0].dtype) if parameters else None,
            device=str(native.device),
            modules=[type(module).__name__ for module in native],
            torch_threads=torch.get_num_threads(),
            parallelism="torch intra-op threads; requested workers are not passed to ST",
        )
    elif cell["backend"] == "fse":
        record.update(
            table_dtype=str(getattr(getattr(native, "word_vectors", None), "dtype", None)),
            parallelism="fse workers",
        )
    else:
        record.update(parallelism="model2vec without multiprocessing")
    return record


def encode_probe(cell, native):
    texts = ["Ein kurzer Satz.", "A short sentence."]
    if cell["backend"] == "fse":
        if native.tokenizer == "external":
            from hub_classic_corpus import tokenize

            texts = [tokenize(text) for text in texts]
        return native.encode(texts, workers=1)
    if cell["backend"] == "model2vec":
        return native.encode(texts, use_multiprocessing=False)
    return native.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def model_files(cell):
    """sha256 of every file of the pinned snapshot in the local cache (weights, tokenizer and
    configuration all come from the one pinned revision)."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    folder = Path(
        snapshot_download(cell["model"], revision=MODEL_PINS[cell["model"]], local_files_only=True)
    )
    if folder.name != MODEL_PINS[cell["model"]]:
        raise ValueError("cached snapshot is not the pinned revision")
    return {
        str(p.relative_to(folder)): sha256(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    }
