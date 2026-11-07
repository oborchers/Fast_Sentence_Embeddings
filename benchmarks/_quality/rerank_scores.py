"""Reranker scores of candidate pairs and the retrieval metrics of the reranker study."""

from __future__ import annotations

import math
import time

import numpy as np
from _measurement_config import MODEL_PINS

METRICS = ("ndcg_cut_10", "recall_10", "recall_100")
DTYPES = ("float32", "float16")


def reranker(spec, context):
    """A Sentence Transformers cross-encoder at its pinned revision, kept for the next task."""
    import torch
    from sentence_transformers import CrossEncoder

    if spec["model"] not in MODEL_PINS or spec["dtype"] not in DTYPES:
        raise ValueError("reranker must be pinned and float32 or float16")
    key = (spec["model"], spec["dtype"], spec["device"], spec["max_length"])
    cached = context.get("rerank_model")
    if cached and cached[0] == key:
        return cached[1]
    context.pop("rerank_model", None)
    __import__("gc").collect()
    if spec["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("the reranker asks for MPS, which this machine lacks")
    model = CrossEncoder(
        spec["model"],
        revision=MODEL_PINS[spec["model"]],
        device=spec["device"],
        max_length=spec["max_length"],
        model_kwargs={"dtype": getattr(torch, spec["dtype"])},
    )
    context["rerank_model"] = (key, model)
    return model


def score_pairs(model, spec, data, pairs):
    """One score per (query, position) pair. Pairs go in order of document length, so a batch
    pads little; the order does not enter the result beyond float rounding of the reranker."""
    ordered = sorted(pairs, key=lambda pair: (-len(data.docs[pair[1]]), pair))
    started = time.perf_counter()
    values = model.predict(
        [(data.queries[q], data.docs[p]) for q, p in ordered],
        batch_size=spec["batch_size"],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    seconds = time.perf_counter() - started
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != len(ordered) or not np.isfinite(values).all():
        raise ValueError("the reranker returned missing or nonfinite scores")
    return dict(zip(ordered, values.tolist(), strict=True)), seconds


def metrics(data, ranking):
    """Mean NDCG@10, Recall@10 and Recall@100 by pytrec_eval, as MTEB computes them. The
    ranking enters as descending scores, so pytrec_eval keeps its order exactly."""
    import pytrec_eval

    run = {}
    for q, row in enumerate(ranking):
        kept = row[row >= 0].tolist()
        run[data.query_ids[q]] = {data.doc_ids[p]: float(len(kept) - r) for r, p in enumerate(kept)}
    evaluator = pytrec_eval.RelevanceEvaluator(data.qrels, {"ndcg_cut.10", "recall.10,100"})
    scores = evaluator.evaluate(run)
    if set(scores) != set(data.query_ids):
        raise ValueError("pytrec_eval dropped queries")
    result = {m: float(np.mean([scores[q][m] for q in data.query_ids])) for m in METRICS}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite retrieval metric")
    return result
