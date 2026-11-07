"""Private native loaders and unmodified MTEB evaluation, without throughput claims."""

from __future__ import annotations

import json
import math
import time
from importlib.metadata import version

from _measurement import atomic, package_identity, sha256
from _measurement_config import MODEL_PINS

from _quality.guard import admit
from _quality.identity import execution, model_files, training_relation
from _quality.suites import filters_for, task_of


def loader(cell, context):

    backend = cell["backend"]
    fit = context["fits"].get(cell.get("fit"))
    key = json.dumps(
        {
            "model": cell["model"],
            "backend": backend,
            "fit": fit,
            "workers": cell["workers"],
            "dtype": cell.get("dtype", "float32"),
            "truncate": cell.get("truncate", False),
            "truncate_dim": cell.get("truncate_dim", "model"),
            "trust_remote_code": cell.get("trust_remote_code", False),
        },
        sort_keys=True,
    )
    if context.get("loaded_key") == key:
        return context["loaded"]
    context.pop("loaded", None)
    __import__("gc").collect()
    context.pop("identity", None)
    model = native_model(cell, fit)
    adapter = make_adapter(cell, model)
    context["identity"] = {
        "execution": execution(cell, model),
        "model_files": None if fit else model_files(cell),
    }
    context["loaded_key"], context["loaded"] = key, adapter
    return adapter


def native_model(cell, fit):
    from pathlib import Path

    import fse

    backend = cell["backend"]
    if cell.get("fit") and fit is None:
        raise ValueError("fit prerequisite did not complete successfully")
    if fit:
        if sha256(Path(fit["artifact"])) != fit["artifact_sha256"]:
            raise ValueError("fitted artifact changed before evaluation")
        model = fse.Encoder.load(fit["artifact"], mmap=True)
    elif backend == "fse":
        if cell["model"].startswith("minishlab/"):
            model = fse.StaticModel.from_pretrained(
                cell["model"], revision=MODEL_PINS[cell["model"]], normalize=True
            ).encoder
        else:
            model = fse.Encoder.from_pretrained(
                cell["model"],
                revision=MODEL_PINS[cell["model"]],
                normalize=True,
                truncate=cell.get("truncate", False),
                truncate_dim=cell.get("truncate_dim", "model"),
            )
    elif backend == "model2vec":
        from _quality.identity import model2vec_pinned

        model = model2vec_pinned(cell["model"])
    elif backend == "st":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            source(cell), revision=MODEL_PINS[cell["model"]], device="cpu", **remote(cell)
        )
    else:
        raise ValueError(f"unknown quality backend: {backend}")
    if backend == "fse" and cell.get("dtype", "float32") != "float32":
        model = model.quantize(cell["dtype"])
    return model


def remote(cell):
    """``trust_remote_code`` only where the cell says so (a real boolean, checked by
    ``validate_cell``)."""
    return {"trust_remote_code": True} if cell.get("trust_remote_code") else {}


def source(cell):
    """What Sentence Transformers loads: the model name, or for a model with its own code the
    local folder of the pinned snapshot. The code of such a model (SSE.py) reads its tokenizer
    and weights without the revision, so only a folder holds every file at the pin."""
    if not cell.get("trust_remote_code"):
        return cell["model"]
    from huggingface_hub import snapshot_download

    return snapshot_download(cell["model"], revision=MODEL_PINS[cell["model"]])


def make_adapter(cell, model):
    from hub_classic_corpus import tokenize

    backend = cell["backend"]
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.models.model_meta import ModelMeta, ScoringFunction

    class Adapter(AbsEncoder):
        def __init__(self):
            self.native = model
            self.mteb_model_meta = ModelMeta.create_empty().model_copy(
                update={
                    "name": cell["model"],
                    "revision": MODEL_PINS[cell["model"]],
                    "similarity_fn_name": ScoringFunction.COSINE,
                    "use_instructions": False,
                }
            )

        def encode(self, inputs, **_kwargs):
            texts = [text for batch in inputs for text in batch["text"]]
            if backend == "fse":
                if model.tokenizer == "external":
                    texts = [tokenize(text) for text in texts]
                return model.encode(texts, workers=cell["workers"])
            if backend == "model2vec":
                return model.encode(texts, use_multiprocessing=False)
            return model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )

    return Adapter()


def evaluate(cell, context):
    package = {"fse": "fse", "model2vec": "model2vec", "st": "sentence-transformers"}[
        cell["backend"]
    ]
    before = package_identity(package)
    model = loader(cell, context)
    task = task_of(cell["benchmark"], cell["task"])
    if filters_for(task) != cell["evaluation_filters"]:
        raise ValueError("configured task filters changed after expansion")
    kept = False
    try:
        estimate = admit(
            task, context["identity"]["execution"]["output_dim"], context.get("rss_limit_bytes")
        )
        record = scored(cell, context, model, task)
        # A local run in task order keeps the data for the next cell of the same task.
        kept = context.get("next_task") == (cell["benchmark"], cell["task"])
    finally:
        if not kept:
            release(task)
    if package_identity(package) != before:
        raise ValueError("executing quality backend changed")
    return {
        **record,
        "status": "valid",
        "benchmark": cell["benchmark"],
        "model_revision": MODEL_PINS[cell["model"]],
        "backend_identity": before,
        "mteb_version": version("mteb"),
        "table_dtype": stored_dtype(cell, model.native),
        "output_dim": getattr(model.native, "output_dim", None),
        "normalization": True,
        "timing_scope": "diagnostic only, never inference throughput",
        "fit_id": cell.get("fit"),
        "fit": context["fits"].get(cell.get("fit")),
        **context["identity"],
        "training_relation": training_relation(cell["model"], task.metadata.name),
        "memory_estimate_bytes": estimate,
    }


def release(task):
    """Free the data of ``task``, also after a refusal or a partly failed load. MTEB unloads
    only data it loaded itself, and the benchmark instance is cached by MTEB, so without this
    every task's data stays in memory."""
    if task.data_loaded:
        task.unload_data()
    elif getattr(task, "dataset", None) is not None:
        task.dataset = None
        task.__dict__.pop("metadata", None)  # as unload_data: back to the published metadata


def scored(cell, context, model, task):
    """Load, evaluate and record one task. Everything read from the task (metadata, filters)
    is taken before its data is unloaded, which restores the unfiltered metadata."""
    import mteb

    load_started = time.perf_counter()
    reused = task.data_loaded
    if not reused:
        task.load_data()
    load_seconds = time.perf_counter() - load_started
    started = time.perf_counter()
    result = mteb.evaluate(
        model,
        [task],
        cache=None,
        overwrite_strategy="always",
        show_progress_bar=False,
        co2_tracker=False,
    )
    if len(result.task_results) != 1:
        raise ValueError("expected exactly one result for the requested task")
    value = result.task_results[0].get_score()
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("MTEB did not produce a finite score")
    if filters_for(task) != cell["evaluation_filters"]:
        raise ValueError("task filters changed during evaluation")
    raw = context["results"] / (cell["id"] + "-mteb.json")
    atomic(raw, result.task_results[0].model_dump(mode="json"))
    return {
        "main_score": value,
        "task": task.metadata.name,
        "task_metadata": task.metadata.model_dump(mode="json"),
        "evaluation_filters": filters_for(task),
        "raw_result_sha256": sha256(raw),
        "load_seconds": load_seconds,
        "data_reused": reused,
        "evaluation_seconds": time.perf_counter() - started,
    }


def stored_dtype(cell, model):
    """The dtype the table is stored in: a model that stores float16 (bekko) says so."""
    table = getattr(model, "word_vectors", None)
    return table.dtype if table is not None else cell.get("dtype", "float32")
