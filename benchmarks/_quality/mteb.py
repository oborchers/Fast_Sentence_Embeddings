"""Private native loaders and unmodified MTEB evaluation, without throughput claims."""

from __future__ import annotations

import json
import math
import time
from importlib.metadata import version

from _measurement import atomic, package_identity, sha256
from _measurement_config import MODEL_PINS, PINS

MIND_REASON = "not computed: MTEB 2.26.25 MindSmallReranking needs more memory than the 64 GB machine (two attempts, status 137 and earlier ENOSPC)"


def tasks_for(suite):
    import mteb

    if version("mteb") != PINS["versions"]["mteb"]:
        raise ValueError("MTEB differs from the pinned evaluator")
    if suite in ("MTEB(eng, v2)", "NanoBEIR"):
        tasks = list(mteb.get_benchmark(suite).tasks)
        expected = 41 if suite == "MTEB(eng, v2)" else 13
        if len(tasks) != expected:
            raise ValueError("pinned benchmark task count changed")
        return [t for t in tasks if t.metadata.name != "MindSmallReranking"]
    task = mteb.get_task(suite)
    if task.metadata.name == "MindSmallReranking":
        raise ValueError(MIND_REASON)
    return [task]


def expand(cells):
    result = []
    membership = {
        t.metadata.name: (suite, t)
        for suite in ("MTEB(eng, v2)", "NanoBEIR")
        for t in tasks_for(suite)
    }
    for cell in cells:
        if cell["kind"] != "mteb":
            result.append(cell)
            continue
        for task, benchmark in cell_tasks(cell, membership):
            result.append(
                {
                    **cell,
                    "id": cell["id"] + "_" + task.metadata.name.replace(".", "-"),
                    "task": task.metadata.name,
                    "benchmark": benchmark,
                    "evaluation_filters": filters_for(task),
                }
            )
    if len({c["id"] for c in result}) != len(result):
        raise ValueError("task expansion produced duplicate identifiers")
    return result


def cell_tasks(cell, membership):
    for suite in cell["tasks"]:
        selected = [membership[suite][1]] if suite in membership else tasks_for(suite)
        for task in selected:
            yield (
                task,
                membership[task.metadata.name][0] if task.metadata.name in membership else suite,
            )


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
        },
        sort_keys=True,
    )
    if context.get("loaded_key") == key:
        return context["loaded"]
    context.pop("loaded", None)
    __import__("gc").collect()
    model = native_model(cell, fit)
    adapter = make_adapter(cell, model)
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
                cell["model"], revision=MODEL_PINS[cell["model"]], normalize=True, truncate=False
            )
    elif backend == "model2vec":
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(
            cell["model"], revision=MODEL_PINS[cell["model"]], normalize=True
        )
    elif backend == "st":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cell["model"], revision=MODEL_PINS[cell["model"]], device="cpu")
    else:
        raise ValueError(f"unknown quality backend: {backend}")
    if backend == "fse" and cell.get("dtype", "float32") != "float32":
        model = model.quantize(cell["dtype"])
    return model


def make_adapter(cell, model):
    from hub_classic_corpus import tokenize

    backend = cell["backend"]
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.models.model_meta import ModelMeta, ScoringFunction

    class Adapter(AbsEncoder):
        def __init__(self):
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
    import mteb

    package = {"fse": "fse", "model2vec": "model2vec", "st": "sentence-transformers"}[
        cell["backend"]
    ]
    before = package_identity(package)
    model = loader(cell, context)
    candidates = tasks_for(cell["benchmark"])
    task = next(t for t in candidates if t.metadata.name == cell["task"])
    if filters_for(task) != cell["evaluation_filters"]:
        raise ValueError("configured task filters changed after expansion")
    load_started = time.perf_counter()
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
    if package_identity(package) != before:
        raise ValueError("executing quality backend changed")
    raw = context["results"] / (cell["id"] + "-mteb.json")
    atomic(raw, result.task_results[0].model_dump(mode="json"))
    return {
        "status": "valid",
        "main_score": value,
        "task": task.metadata.name,
        "task_metadata": task.metadata.model_dump(mode="json"),
        "evaluation_filters": filters_for(task),
        "benchmark": cell["benchmark"],
        "model_revision": MODEL_PINS[cell["model"]],
        "backend_identity": before,
        "mteb_version": version("mteb"),
        "table_dtype": cell.get("dtype", "float32"),
        "normalization": True,
        "load_seconds": load_seconds,
        "evaluation_seconds": time.perf_counter() - started,
        "timing_scope": "diagnostic only, never inference throughput",
        "fit_id": cell.get("fit"),
        "fit": context["fits"].get(cell.get("fit")),
    }


def filters_for(task):
    return {
        "hf_subsets": list(task.hf_subsets),
        "eval_splits": list(task.eval_splits),
        "seed": task.seed,
    }
