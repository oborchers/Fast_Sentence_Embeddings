"""The data of one pinned NanoBEIR task for the reranker study, read as MTEB reads it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskData:
    """Texts, IDs and judgments of one task; positions index ``doc_ids`` and ``docs``."""

    name: str
    revision: str
    doc_ids: list
    docs: list
    query_ids: list
    queries: list
    qrels: dict
    ignore_identical_ids: bool


def _document(row):
    """A document as MTEB encodes it: title and text joined, or the text alone."""
    title = row.get("title")
    return (title + " " + row["text"]).strip() if title else row["text"].strip()


def task_data(name):
    """The pinned task of the NanoBEIR benchmark, with the queries MTEB keeps (those with
    judged documents). Refuses what this study does not reproduce: other subsets or splits,
    a given candidate list, skipped first results, unknown or repeated IDs."""
    from _quality.mteb import tasks_for

    task = next(t for t in tasks_for("NanoBEIR") if t.metadata.name == name)
    if list(task.hf_subsets) != ["default"] or list(task.eval_splits) != ["train"]:
        raise ValueError(f"{name}: unexpected subsets or splits")
    if task.skip_first_result:
        raise ValueError(f"{name}: skip_first_result is not reproduced")
    task.load_data()
    split = task.dataset["default"]["train"]
    if split["top_ranked"] is not None:
        raise ValueError(f"{name}: a task with given candidates is not a first-stage task")
    corpus = split["corpus"]
    doc_ids = list(corpus["id"])
    docs = [_document(row) for row in corpus]
    qrels = {q: dict(d) for q, d in split["relevant_docs"].items() if len(d)}
    kept = [
        (i, t)
        for i, t in zip(split["queries"]["id"], split["queries"]["text"], strict=True)
        if i in qrels
    ]
    if len(set(doc_ids)) != len(doc_ids) or len({i for i, _ in kept}) != len(kept):
        raise ValueError(f"{name}: repeated document or query IDs")
    if not set().union(*qrels.values()) <= set(doc_ids) or {i for i, _ in kept} != set(qrels):
        raise ValueError(f"{name}: judgments name unknown documents or queries")
    return TaskData(
        name=name,
        revision=task.metadata.dataset["revision"],
        doc_ids=doc_ids,
        docs=docs,
        query_ids=[i for i, _ in kept],
        queries=[t for _, t in kept],
        qrels=qrels,
        ignore_identical_ids=bool(task.ignore_identical_ids),
    )
