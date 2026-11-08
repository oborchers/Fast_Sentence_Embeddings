"""Private identity of an evaluated model: training relation, execution path and files.

These records explain a score; they never change it. They are read once per loaded model,
except the training relation, which is per task."""

from __future__ import annotations

import sys

from _harness.records import sha256
from _measurement_config import MODEL_PINS

UNKNOWN, OVERLAP, ZERO_SHOT = "unknown", "training_overlap", "zero_shot"
# Registry revisions of MTEB 2.26.25 whose declared training data holds for the pin, with
# the evidence. Any other difference between registry revision and pin gives "unknown".
EQUIVALENT = {
    (
        "sentence-transformers/static-similarity-mrl-multilingual-v1",
        "7264ea07c5365a11d7e6d87dbb6195889a13054f",
        "b68f4122911bcffcd6e1f695f2d99cd6788972d8",
    ): "every file but README.md has the same Hub object id in both revisions (07.11.2026)",
}


def transferable(model, registry_revision):
    """Whether the training declaration of ``registry_revision`` holds for the pin."""
    pin = MODEL_PINS[model]
    return registry_revision == pin or (model, registry_revision, pin) in EQUIVALENT


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
    if not transferable(model, meta.revision):
        return {**record, "state": UNKNOWN, "reason": "registry revision is not the pin"}
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


def model2vec_pinned(model):
    """A Model2Vec model at its pin: model2vec 0.10.0 takes no revision, so it loads the
    folder of the pinned snapshot."""
    from huggingface_hub import snapshot_download
    from model2vec import StaticModel

    folder = snapshot_download(model, revision=MODEL_PINS[model])
    return StaticModel.from_pretrained(folder, normalize=True, force_download=False)


def model_files(cell):
    """sha256 of every file of the pinned snapshot in the local cache (weights, tokenizer and
    configuration all come from the one pinned revision). Only the files a loader fetched
    are there; the snapshot need not hold the whole repository."""
    from pathlib import Path

    from huggingface_hub import constants

    pin = MODEL_PINS[cell["model"]]
    folder = (
        Path(constants.HF_HUB_CACHE)
        / ("models--" + cell["model"].replace("/", "--"))
        / "snapshots"
        / pin
    )
    if not folder.is_dir():
        raise ValueError("the pinned snapshot is not in the local cache")
    return {
        str(p.relative_to(folder)): sha256(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    }
