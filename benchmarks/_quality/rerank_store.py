"""Exact persisted-output quality arms, sharing the reranker study and MTEB data."""

from __future__ import annotations

import gc
import os
from contextlib import ExitStack
from importlib.metadata import version
from tempfile import TemporaryDirectory

import numpy as np
from _measurement import package_identity
from _measurement_config import MODEL_PINS, validate_cell

from _quality import rerank_scores, rerank_stages
from _quality.guard import RefusedError, resident_bytes
from _quality.rerank_data import task_data
from _quality.suites import filters_for, task_of

DTYPES = ("float32", "float16", "int8")
DEPTHS = (10, 20, 50, 100, 200, 500, 1000)


def validate(cell, stages):
    """Refuse incompatible tables, missing references and ambiguous format configurations."""
    selected = [stages[n] for n in cell["stages"]]
    if not any(s["kind"] in ("store", "rescore") for s in selected):
        return
    if cell.get("reranker") or cell.get("depths"):
        raise ValueError("store quality cells cannot contain a cross-encoder")
    if set(cell.get("mteb_check", [])) != {s["encoder"] for s in selected}:
        raise ValueError("every store encoder needs its dense MTEB check")
    encoders = {s["encoder"] for s in selected}
    if len(encoders) != 1:
        raise ValueError("one store cell evaluates exactly one encoder")
    if sum(s["kind"] == "store" and s.get("dtype") == "float32" for s in selected) != 1:
        raise ValueError("one float32 store reference required")
    if len(cell["stages"]) != len(set(cell["stages"])):
        raise ValueError("duplicate store stage")
    for name in cell["stages"]:
        validate_cell({"id": name})
        _validate_stage(stages[name], stages, cell["stages"])


def _validate_stage(stage, stages, selected):
    if stage["kind"] not in ("store", "rescore"):
        raise ValueError("store cells select only store and rescore arms")
    encoder = stages.get(stage["encoder"], {})
    validate_cell({"id": stage["encoder"], **encoder})
    if (encoder.get("kind"), encoder.get("backend"), encoder.get("dtype", "float32")) != (
        "dense",
        "fse",
        "float32",
    ) or encoder.get("model") not in MODEL_PINS:
        raise ValueError("store stages need a pinned dense fse float32 table")
    if type(encoder.get("workers")) is not int or encoder["workers"] <= 0:
        raise ValueError("store encoder workers must be a positive integer")
    if stage["kind"] == "store":
        if stage.get("dtype") not in DTYPES:
            raise ValueError("store dtype must be float32, float16 or int8")
    else:
        _validate_rescore(stage, stages, selected)


def _validate_rescore(stage, stages, selected):
    for key, dtype in (("input", "int8"), ("reference", "float32")):
        other = stages.get(stage.get(key), {})
        if stage.get(key) not in selected or (
            other.get("kind"),
            other.get("dtype"),
            other.get("encoder"),
        ) != ("store", dtype, stage["encoder"]):
            raise ValueError("rescore needs selected int8 and float32 stores of the same encoder")
    depths = stage.get("depths")
    if not isinstance(depths, list) or any(type(d) is not int for d in depths):
        raise ValueError("rescore depths must be an ordered integer list")
    if tuple(depths) != DEPTHS:
        raise ValueError(f"rescore depths must be {DEPTHS}")


def admit(data, dim, limit):
    """Bound full dense scores, sorting scratch, two decoded formats and text copies.

    This second admission uses actual full data sizes, before encoding or allocating
    score matrices. MTEB's metadata guard runs before loading the task itself.
    """
    phases = memory_estimate(data, dim)
    estimate = max(phases.values())
    left = None if limit is None else limit - (resident_bytes(os.getpid()) or 0)
    if left is not None and estimate > left:
        raise RefusedError(f"retrieval memory estimate {estimate} bytes exceeds {left} bytes left")
    return {"peak_bytes": estimate, "phases": phases}


def memory_estimate(data, dim):
    """Conservative phase estimates, including the pinned MTEB search implementation.

    MTEB 2.26.25 SearchEncoderWrapper scores chunks of at most 50,000 documents,
    holding the study's dense matrix concurrently. Its normalized vector copies,
    topk tensors, Python heaps, result dictionaries and evaluator copies also overlap.
    These are admission estimates, not measured peak RSS guarantees.
    """
    q, d = len(data.queries), len(data.docs)
    k, chunk = min(d, rerank_stages.DEPTH), min(d, 50_000)
    characters = sum(map(len, data.docs)) + sum(map(len, data.queries))
    text_scratch = 8 * characters + (d + q) * 512
    retained_dense = 4 * q * d + 8 * q * k
    dense = 4 * q * d + 32 * d + 8 * (d + q) * dim + 256 * q * k
    mteb = retained_dense + 4 * q * chunk + 8 * (chunk + q) * dim + 512 * q * k
    stores = retained_dense + 8 * (d + q) * dim + 128 * q * k
    return {
        "dense": text_scratch + dense,
        "mteb_check": text_scratch + mteb,
        "stores_and_rescore": text_scratch + stores,
    }


def store(encoder, data, dtype, folder, workers):
    """Encode corpus order into a fresh ShardSink, with positional document IDs."""
    import fse

    positions = np.arange(len(data.docs), dtype=np.int64)
    with fse.ShardSink(folder, encoder, dtype=dtype, doc_ids=True) as sink:
        encoder.encode(data.docs, sink=sink, doc_ids=positions, workers=workers)
    result = fse.VectorStore.open(folder, encoder=encoder)
    try:
        if len(result) != len(data.doc_ids) or result.dtype != dtype:
            raise ValueError("store row count or dtype differs from the corpus")
        if not np.array_equal(result.get_doc_ids(), positions):
            raise ValueError("store document IDs differ from corpus positions")
        result.verify()
    except BaseException:
        result.close()
        raise
    details = {
        "rows": len(result),
        "dtype": result.dtype,
        "bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
        "encoder_fingerprint": encoder.fingerprint,
        "doc_ids": "corpus positions, indexing the saved doc_ids array",
    }
    return result, details


def agreement(reference, ranking):
    return float(np.mean(np.all(reference[:, :10] == ranking[:, :10], axis=1)))


def rescore(queries, candidates, reference, baseline, data, depths, workers):
    """Pure candidate lists. No first-stage tail is appended to a rescore arm."""
    arms, comparisons, rankings = {}, {}, {}
    for depth in depths:
        kept = candidates[:, :depth]
        hit = reference.rescore(queries, kept, k=depth, workers=workers)
        covered = np.array(
            [np.isin(row[:10], pool).mean() for row, pool in zip(baseline, kept, strict=True)]
        )
        ordered = np.all(baseline[:, :10] == hit.ids[:, :10], axis=1)
        equal = float(np.mean(ordered))
        full = float(np.mean(covered == 1))
        if not np.array_equal(ordered, covered == 1):
            raise ValueError("ordered Top-10 agreement differs from full candidate coverage")
        name = f"int8>f32@{depth} (candidates)"
        arms[name] = rerank_scores.metrics(data, hit.ids)
        comparisons[name] = {
            "depth": depth,
            "coverage_float32_top10": float(covered.mean()),
            "full_coverage_queries": full,
            "ordered_top10_equal": equal,
            "list_shorter_than_100": hit.ids.shape[1] < 100,
        }
        rankings[name] = hit.ids
    return arms, comparisons, rankings


def smallest(arms, comparisons, baseline):
    rules = {
        "recall_10": lambda n, _c: arms[n]["recall_10"] >= baseline["recall_10"],
        "recall_100": lambda n, c: (
            c["depth"] >= 100 and arms[n]["recall_100"] >= baseline["recall_100"]
        ),
        "full_float32_top10_coverage": lambda _n, c: c["coverage_float32_top10"] == 1,
    }
    return {
        key: next((c["depth"] for n, c in comparisons.items() if rule(n, c)), None)
        for key, rule in rules.items()
    }


def _direct(cell, stages, adapter, data, queries, folder, stack, dense_ranking, dense_scores):
    stores, rankings, arms, details, comparisons = {}, {}, {}, {}, {}
    workers = stages[stages[cell["stages"][0]]["encoder"]]["workers"]
    ordered = sorted(cell["stages"], key=lambda n: stages[n].get("dtype") != "float32")
    for name in ordered:
        stage = stages[name]
        if stage["kind"] != "store":
            continue
        opened, details[name] = store(adapter.native, data, stage["dtype"], folder / name, workers)
        with opened:
            hit = opened.search(queries, rerank_stages.DEPTH, mode="exact", workers=workers)
            rankings[name] = hit.ids
            arms[name] = rerank_scores.metrics(data, hit.ids)
            if stage["dtype"] == "float32":
                import fse

                stores[name] = stack.enter_context(fse.VectorStore.open(folder / name))
                reference = hit.ids
                details[name]["numpy_comparison"] = {
                    "top10_different_positions": int(
                        np.count_nonzero(dense_ranking[:, :10] != hit.ids[:, :10])
                    ),
                    "max_score_difference_on_store_top1000": float(
                        np.max(
                            np.abs(np.take_along_axis(dense_scores, hit.ids, axis=1) - hit.scores)
                        )
                    ),
                    "meaning": "arithmetic path difference, not output quantization error",
                }
                del dense_scores, dense_ranking
            comparisons[name] = {"ordered_top10_equal": agreement(reference, hit.ids)}
    return stores, rankings, arms, details, comparisons


def _model(cell, stages, data, context, folder):
    from _quality.mteb import loader
    from _quality.rerank import _mteb_check

    (encoder_name,) = {stages[n]["encoder"] for n in cell["stages"]}
    spec = stages[encoder_name]
    adapter = loader(spec, context)
    estimate = admit(data, adapter.native.output_dim, context.get("rss_limit_bytes"))
    dense_ranking, dense_scores = rerank_stages.dense(adapter, data)
    key = (data.benchmark, data.name, data.hf_subset, data.split, encoder_name)
    context.setdefault("rerank_raw", {})[key] = dense_scores
    dense_metrics = {encoder_name: rerank_scores.metrics(data, dense_ranking)}
    checks = _mteb_check(cell, stages, data, context, dense_metrics)
    context["rerank_raw"].pop(key)
    queries = adapter.native.encode(data.queries, workers=spec["workers"])
    with ExitStack() as stack:
        stores, rankings, arms, details, comparisons = _direct(
            cell, stages, adapter, data, queries, folder, stack, dense_ranking, dense_scores
        )
        del dense_ranking, dense_scores
        depths = {}
        for name in cell["stages"]:
            stage = stages[name]
            if stage["kind"] != "rescore":
                continue
            reference_name = stage["reference"]
            a, c, r = rescore(
                queries,
                rankings[stage["input"]],
                stores[reference_name],
                rankings[reference_name],
                data,
                stage["depths"],
                spec["workers"],
            )
            depths[name] = smallest(a, c, arms[reference_name])
            # Prefix with the configured stage name so models never share an arm.
            arms.update({f"{name}/{n}": v for n, v in a.items()})
            comparisons.update({f"{name}/{n}": v for n, v in c.items()})
            rankings.update({f"{name}/{n}": v for n, v in r.items()})
        np.savez_compressed(
            context["results"] / (cell["id"] + "-rankings.npz"),
            doc_ids=np.array(data.doc_ids),
            query_ids=np.array(data.query_ids),
            **rankings,
        )
    return {
        "arms": arms,
        "comparisons": comparisons,
        "smallest_tested_depth": depths,
        "unreached_depth": "not reached through 1000",
        "stores": details,
        "mteb_check": checks,
        "memory_estimate_bytes": estimate["peak_bytes"],
        "memory_estimate_phases": estimate["phases"],
        "model_revision": MODEL_PINS[spec["model"]],
        "table_dtype": "float32",
        "output_dim": adapter.native.output_dim,
        "execution": context["identity"],
    }


def evaluate(cell, context):
    from _quality.guard import admit as admit_task
    from _quality.mteb import release
    from _quality.rerank import _sources

    stages = context["study"]["first_stages"]
    validate(cell, stages)
    task = task_of(cell["benchmark"], cell["task"])
    if filters_for(task) != cell["evaluation_filters"]:
        raise ValueError("configured task filters changed after expansion")
    before = package_identity("fse")
    try:
        if task.ignore_identical_ids:
            raise RefusedError(
                "store quality does not support ignore_identical_ids=True; FiQA2018 is deferred"
            )
        # The largest model in this study is 1024-dimensional. Actual admission follows loading.
        admit_task(task, 1024, context.get("rss_limit_bytes"))
        data = task_data(cell["benchmark"], cell["task"])
        if len(data.docs) < 10:
            raise ValueError("store quality requires at least ten corpus documents")
        with TemporaryDirectory(prefix="retrieval-", dir=context["work"]) as folder:
            from pathlib import Path

            result = _model(cell, stages, data, context, Path(folder))
        if package_identity("fse") != before:
            raise ValueError("executing fse changed")
        return {
            **result,
            "status": "valid",
            "task": data.name,
            "benchmark": data.benchmark,
            "hf_subset": data.hf_subset,
            "split": data.split,
            "dataset_revision": data.revision,
            "queries": len(data.queries),
            "documents": len(data.docs),
            "search_mode": "exact",
            "backend_identity": before,
            "versions": {p: version(p) for p in ("fse", "mteb", "numpy", "pytrec_eval-terrier")},
            "first_stages": {n: stages[n] for n in cell["stages"] + cell["mteb_check"]},
            "study_sources": _sources(),
        }
    finally:
        context.pop("rerank_raw", None)
        release(task)
        gc.collect()
