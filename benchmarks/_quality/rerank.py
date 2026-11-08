"""Cells of kind ``rerank``: first stages and local rerankers on NanoBEIR, class K.

A cell names first stages from the study's ``first_stages`` and optionally one reranker with
candidate depths. It expands into one cell per NanoBEIR task like an MTEB cell. Rankings of a
task are computed once per run and shared by the cells; raw rankings and pair scores stay in
the result folder. Reranker seconds are a local diagnostic, never a speed result.
"""

from __future__ import annotations

import time
from importlib.metadata import version

import numpy as np
from _measurement import package_identity
from _measurement_config import MODEL_PINS

from _quality import rerank_scores, rerank_stages
from _quality.rerank_data import task_data

KINDS = ("dense", "bm25", "rrf")
PACKAGES = ("mteb", "bm25s", "PyStemmer", "pytrec_eval-terrier", "transformers", "torch", "numpy")


def _sources():
    """SHA-256 of the study modules that computed the row."""
    from pathlib import Path

    from _harness.records import sha256

    here = Path(__file__).parent
    # mteb.py evaluates through suites.py (tasks), identity.py and guard.py (admission).
    shared = [here / f"{name}.py" for name in ("mteb", "suites", "identity", "guard")]
    return {p.name: sha256(p) for p in [*sorted(here.glob("rerank*.py")), *shared]}


def expand(cells):
    """One cell per NanoBEIR task; other kinds pass unchanged."""
    from _quality.suites import benchmark_tasks

    result = []
    for cell in cells:
        if cell["kind"] != "rerank":
            result.append(cell)
            continue
        if cell["tasks"] != ["NanoBEIR"]:
            raise ValueError("a rerank cell covers the NanoBEIR benchmark")
        for task in benchmark_tasks("NanoBEIR"):
            name = task.metadata.name
            result.append({**cell, "id": cell["id"] + "_" + name, "task": name})
    return result


def validate(cell, stages):
    """The cell names known first stages, depths within the rankings, a pinned reranker."""
    for name in cell["stages"] + cell.get("mteb_check", []):
        if name not in stages or stages[name]["kind"] not in KINDS:
            raise ValueError(f"unknown first stage: {name}")
    for name in cell.get("mteb_check", []):
        if stages[name].get("backend") != "fse":
            raise ValueError("the MTEB check covers fse first stages")
    spec = cell.get("reranker")
    depths = cell.get("depths", [])
    if (spec is None) != (not depths) or any(
        not isinstance(d, int) or not 0 < d <= rerank_stages.DEPTH for d in depths
    ):
        raise ValueError("a reranker needs candidate depths between 1 and 1000, and only it")
    if spec is not None and spec["model"] not in MODEL_PINS:
        raise ValueError(f"reranker is not pinned: {spec['model']}")


def _stage(name, stages, data, context, rankings):
    if name in rankings:
        return rankings[name]
    stage = stages[name]
    if stage["kind"] == "bm25":
        rankings[name] = rerank_stages.bm25(data)
    elif stage["kind"] == "rrf":
        inputs = [_stage(n, stages, data, context, rankings) for n in stage["inputs"]]
        rankings[name] = rerank_stages.rrf(inputs)
    else:
        from _quality.mteb import loader

        rankings[name], scores = rerank_stages.dense(loader(stage, context), data)
        context.setdefault("rerank_raw", {})[(data.name, name)] = scores
    return rankings[name]


MTEB_KEYS = {
    "ndcg_cut_10": "ndcg_at_10",
    "recall_10": "recall_at_10",
    "recall_100": "recall_at_100",
}


def _mteb_check(cell, stages, data, context, arms):
    """Unmodified MTEB on the same task and encoder. The study's raw cosine scores, handed to
    pytrec_eval as MTEB hands them and rounded as MTEB rounds, must give exactly MTEB's
    NDCG@10, Recall@10 and Recall@100: same data, texts and vectors. The study ranking
    breaks exact ties by position, pytrec_eval by document ID, so its values are recorded
    beside them (they differ only where documents tie exactly, as duplicates in ArguAna)."""
    import json

    from _quality.mteb import evaluate
    from _quality.suites import benchmark_tasks, filters_for

    task = next(t for t in benchmark_tasks("NanoBEIR") if t.metadata.name == data.name)
    checks = {}
    for name in cell.get("mteb_check", []):
        mteb_cell = {
            **stages[name],
            "id": f"{cell['id']}-mteb-{name}",
            "task": data.name,
            "benchmark": "NanoBEIR",
            "evaluation_filters": filters_for(task),
        }
        evaluate(mteb_cell, context)
        raw = json.loads((context["results"] / (mteb_cell["id"] + "-mteb.json")).read_text())
        (split,) = raw["scores"]["train"]
        mteb = {m: split[key] for m, key in MTEB_KEYS.items()}
        study = rerank_scores.scored_metrics(data, context["rerank_raw"][(data.name, name)])
        checks[name] = {"mteb": mteb, "study_scores": study, "study_ranking": arms[name]}
        if study != mteb:
            raise ValueError(f"{name} on {data.name}: study metrics differ from MTEB")
    return checks


def _reranked(cell, data, context, rankings, results):
    spec = cell["reranker"]
    pairs = set()
    for name in cell["stages"]:
        pairs |= rerank_stages.candidates(rankings[name], max(cell["depths"]))
    model = rerank_scores.reranker(spec, context)
    scores, seconds = rerank_scores.score_pairs(model, spec, data, sorted(pairs))
    keys = sorted(scores)
    np.savez_compressed(
        results / (cell["id"] + "-pairs.npz"),
        query=np.array([k[0] for k in keys], np.int32),
        position=np.array([k[1] for k in keys], np.int32),
        score=np.array([scores[k] for k in keys], np.float64),
    )
    short = spec["model"].split("/")[-1]
    arms = {}
    for name in cell["stages"]:
        for depth in cell["depths"]:
            ranking = rerank_stages.rerank(rankings[name], depth, scores)
            arms[f"{name} > {short} @{depth}"] = rerank_scores.metrics(data, ranking)
    return arms, {
        "reranker": {**spec, "revision": MODEL_PINS[spec["model"]]},
        "pairs": len(keys),
        "diagnostic_seconds_per_query": seconds / len(data.queries),
        "timing_scope": "local diagnostic of the reranker only, never a speed result",
    }


def evaluate(cell, context):
    stages = context["study"]["first_stages"]
    validate(cell, stages)
    started = time.perf_counter()
    data = task_data(cell["task"])
    rankings = context.setdefault("rerank_rankings", {}).setdefault(cell["task"], {})
    for name in cell["stages"] + cell.get("mteb_check", []):
        _stage(name, stages, data, context, rankings)
    np.savez_compressed(
        context["results"] / (cell["id"] + "-rankings.npz"),
        **{name: rankings[name].astype(np.int32) for name in cell["stages"]},
    )
    arms = {name: rerank_scores.metrics(data, rankings[name]) for name in cell["stages"]}
    result = {"mteb_check": _mteb_check(cell, stages, data, context, arms)}
    if cell.get("reranker"):
        reranked, details = _reranked(cell, data, context, rankings, context["results"])
        arms.update(reranked)
        result.update(details)
    return {
        **result,
        "status": "valid",
        "task": data.name,
        "benchmark": "NanoBEIR",
        "dataset_revision": data.revision,
        "queries": len(data.queries),
        "documents": len(data.docs),
        "arms": arms,
        "first_stages": {name: stages[name] for name in cell["stages"]},
        "rrf_k": rerank_stages.RRF_K,
        "fse": package_identity("fse"),
        "sentence_transformers": package_identity("sentence-transformers"),
        "versions": {p: version(p) for p in PACKAGES},
        "study_sources": _sources(),
        "evaluation_seconds": time.perf_counter() - started,
    }
