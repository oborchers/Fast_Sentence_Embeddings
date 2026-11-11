"""Small helpers for the reranker recipe: a pinned NanoBEIR task, NDCG@10, and RRF."""

import math

from huggingface_hub import snapshot_download

SCIFACT = ("mteb/NanoSciFactRetrieval", "8c5f7bbe90bb865f46c67710221afa7d9096251b")


def nano_task(recipe, repo=SCIFACT[0], revision=SCIFACT[1]):
    """Documents, queries and judgments of one NanoBEIR task at a pinned revision.

    Returns ``docs`` (document ID to text, title and text joined as MTEB does),
    ``queries`` (query ID to text, only queries with judged documents) and
    ``qrels`` (query ID to {document ID: relevance}).
    """

    import pyarrow.parquet as pq

    def table(name):
        path = recipe.source_file(
            repo, revision, f"{name}/train-00000-of-00001.parquet", repo_type="dataset"
        )
        return pq.read_table(path).to_pylist()

    docs = {}
    for row in table("corpus"):
        title = row.get("title") or ""
        docs[row["id"]] = (title + " " + row["text"]).strip() if title else row["text"].strip()
    qrels = {}
    for row in table("qrels"):
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {row["id"]: row["text"] for row in table("queries") if row["id"] in qrels}
    return docs, queries, qrels


def ndcg_at_10(ranked, relevant):
    """NDCG@10 of one ranked list of document IDs, with graded relevance as the gain."""
    gain = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(relevant.values(), reverse=True)[:10]
    best = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return gain / best if best else 0.0


def mean_ndcg(rankings, qrels):
    """Mean NDCG@10 over the queries of ``rankings`` (query ID to ranked document IDs)."""
    return sum(ndcg_at_10(r, qrels[q]) for q, r in rankings.items()) / len(rankings)


def rrf(*lists, k=60):
    """Reciprocal Rank Fusion of ranked lists: sum of 1 / (k + rank), rank from one.

    k = 60 is the constant of Cormack, Clarke and Buettcher (SIGIR 2009).
    Ties keep the order of first appearance.
    """
    score = {}
    for ranked in lists:
        for rank, doc in enumerate(ranked, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda d: -score[d])


def snapshot(recipe, repo, revision):
    """Download a complete pinned model into the recipe's temporary cache.

    A shared Hub snapshot may hold only metadata, so a local-only lookup does not prove
    that a model is ready to load. Let the Hub finish the download in our own workspace.
    """
    return snapshot_download(repo, revision=revision, cache_dir=recipe.work / "hub", token=False)
