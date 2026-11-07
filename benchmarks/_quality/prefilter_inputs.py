"""Profile int8-prefilter@1: cell contract, query corpora and the WikiText training store texts."""

from __future__ import annotations

PROFILE = "int8-prefilter@1"
SEARCH_PROFILE = "int8-prefilter-search@1"  # the speed cells of DR-278 (benchmarks/_development_prefilter_search.py)
STAGE = "prefilter-count"
TRAIN = "wikitext-train"
QUERY_CORPORA = ("stsb", "wikitext")
TRAIN_ROWS = 859_346  # the paragraphs of hub_classic_corpus.fit_corpus, before tokenizing
SCOPES = ("main", "probe")


def enabled(cell):
    return cell.get("profile") == PROFILE


def search_enabled(cell):
    return cell.get("profile") == SEARCH_PROFILE


def corpus_names(cell):
    """The prepared corpora a cell reads: its own plus the stand-in query corpora (the
    search cells of DR-278 read STS-B only)."""
    if search_enabled(cell):
        return (cell["corpus"], "stsb")
    return (cell["corpus"], *QUERY_CORPORA) if enabled(cell) else (cell["corpus"],)


def _positive_list(cell, name):
    values = cell[name]
    if (
        not isinstance(values, list)
        or not values
        or any(type(v) is not int or v <= 0 for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must be a list of distinct positive integers")


def validate(cell):
    if not enabled(cell) and (
        cell["stage"] == STAGE or (cell["corpus"] == TRAIN and not search_enabled(cell))
    ):
        raise ValueError(f"the stage {STAGE} and the corpus {TRAIN} belong to {PROFILE}")
    if not enabled(cell):
        return
    if cell["stage"] != STAGE or cell["corpus"] != TRAIN or cell["scope"] not in SCOPES:
        raise ValueError(
            f"{PROFILE} needs the stage {STAGE}, the corpus {TRAIN}, scope main or probe"
        )
    if cell["table_dtype"] != "float32" or cell["decision_gate"]:
        raise ValueError(f"{PROFILE} counts rows of an int8 store from a float32 table, no gate")
    _sizes(cell)


def _sizes(cell):
    for name in ("store_sizes", "k_values"):
        _positive_list(cell, name)
    for name in ("queries_per_set", "probe_queries"):
        if type(cell[name]) is not int or cell[name] < 8 or cell[name] % 8:
            raise ValueError(f"{name} must be a multiple of eight, at least eight")
    if type(cell["probe_rows"]) is not int or cell["probe_rows"] < 70_000:
        raise ValueError("probe_rows must reach past the block edge 65,536 (at least 70,000)")
    if cell["scope"] == "main" and max(cell["store_sizes"]) > TRAIN_ROWS:
        raise ValueError("a store cannot be larger than the training corpus")
