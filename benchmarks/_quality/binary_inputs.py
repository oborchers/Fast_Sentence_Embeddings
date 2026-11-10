"""Profiles binary-recall@1 and @2: cell contract and fixed numbers of the DR-255 recall probes (class correctness)."""

from __future__ import annotations

PROFILE = "binary-recall@1"
PROFILE_2 = "binary-recall@2"  # variants A and B, four factors, a float32 and an int8 store
PROFILES = (PROFILE, PROFILE_2)
STAGE = "binary-recall"
CORPUS = "agnews"
AGNEWS = {"repo": "fancyzhx/ag_news", "revision": "eb185aade064a813bc0b7f42de02595523103ca4"}
CORPUS_ROWS = 100_000  # AG News train texts, PCG64 seed 0, sampled without replacement
QUERIES = 1_000  # AG News test texts that are not in the corpus
POOL_QUERIES = 1_500  # candidates for the queries: some may encode to a zero vector
K = 10
OVERSAMPLES = (10, 40)
OVERSAMPLES_2 = (10, 20, 40, 80)
VARIANTS = ("A", "B")  # A: symmetric Hamming; B: asymmetric int4 query codes (plan 3.12)
PROBE_QUERIES = 64  # one block of queries: the run-time probe of @2 before the full run
BOOTSTRAP = 1_000
MODELS = ("minishlab/potion-base-8M", "sentence-transformers/static-retrieval-mrl-en-v1")


def enabled(cell):
    return cell.get("profile") in PROFILES


def validate(cell):
    if not enabled(cell):
        if cell["stage"] == STAGE or cell["corpus"] == CORPUS:
            raise ValueError(f"the stage {STAGE} and the corpus {CORPUS} belong to {PROFILES}")
        return
    profile = cell["profile"]
    if cell["stage"] != STAGE or cell["corpus"] != CORPUS or cell["model"] not in MODELS:
        raise ValueError(
            f"{profile} needs the stage {STAGE}, the corpus {CORPUS} and a probe model"
        )
    if cell["table_dtype"] != "float32" or cell["decision_gate"] or cell["scope"] != "main":
        raise ValueError(f"{profile} reads a float32 table and takes no decision gate")
    queries = (QUERIES, PROBE_QUERIES) if profile == PROFILE_2 else (QUERIES,)
    if (
        cell["store_sizes"] != [CORPUS_ROWS]
        or cell["queries_per_set"] not in queries
        or cell["k_values"] != [K]
    ):
        raise ValueError(
            f"{profile} is fixed at {CORPUS_ROWS} stored texts, {queries} queries and k={K}"
        )
