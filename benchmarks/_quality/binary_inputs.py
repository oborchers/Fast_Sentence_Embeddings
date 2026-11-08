"""Profile binary-recall@1: cell contract and fixed numbers of the DR-255 recall probe (class correctness)."""

from __future__ import annotations

PROFILE = "binary-recall@1"
STAGE = "binary-recall"
CORPUS = "agnews"
AGNEWS = {"repo": "fancyzhx/ag_news", "revision": "eb185aade064a813bc0b7f42de02595523103ca4"}
CORPUS_ROWS = 100_000  # AG News train texts, PCG64 seed 0, sampled without replacement
QUERIES = 1_000  # AG News test texts that are not in the corpus
POOL_QUERIES = 1_500  # candidates for the queries: some may encode to a zero vector
K = 10
OVERSAMPLES = (10, 40)
BOOTSTRAP = 1_000
MODELS = ("minishlab/potion-base-8M", "sentence-transformers/static-retrieval-mrl-en-v1")


def enabled(cell):
    return cell.get("profile") == PROFILE


def validate(cell):
    if not enabled(cell):
        if cell["stage"] == STAGE or cell["corpus"] == CORPUS:
            raise ValueError(f"the stage {STAGE} and the corpus {CORPUS} belong to {PROFILE}")
        return
    if cell["stage"] != STAGE or cell["corpus"] != CORPUS or cell["model"] not in MODELS:
        raise ValueError(
            f"{PROFILE} needs the stage {STAGE}, the corpus {CORPUS} and a probe model"
        )
    if cell["table_dtype"] != "float32" or cell["decision_gate"] or cell["scope"] != "main":
        raise ValueError(f"{PROFILE} reads a float32 store and takes no decision gate")
    if (cell["store_sizes"], cell["queries_per_set"], cell["k_values"]) != (
        [CORPUS_ROWS],
        QUERIES,
        [K],
    ):
        raise ValueError(
            f"{PROFILE} is fixed at {CORPUS_ROWS} stored texts, {QUERIES} queries and k={K}"
        )
