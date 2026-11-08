"""Profile int8-prefilter-search@1: the speed cells of the int8 prefilter in the exact search.

Class optimization (DR-278, plan v2 of DR-168, section 7). The store is the WikiText-103 training
paragraphs (859,346 rows, ``wikitext-train``) as int8 rows of one model; the queries are 256
fixed STS-B test texts that are not in the store (PCG64 seed 0), encoded outside every timer.
Every call of a group searches the next block of ``queries`` of them, round and round, and a
group starts at the first block again, so two builds are compared on the same sequence of
calls and the output of a group is that of its last call. The build that searches does not
change what is searched: the outputs of base and candidate must be bit-identical, which the
paired blocks check for every group (for the ``blas`` search mode that is what makes the cell
runnable at all; an output that differs makes it not runnable and it is reported so).
"""

from __future__ import annotations

from _quality.prefilter_inputs import SEARCH_PROFILE, TRAIN, search_enabled

POOL = 256  # the queries of the pool; a block of ``queries`` of them per call
MODES = ("exact", "blas")


def validate(cell):
    if cell["search_mode"] not in MODES:
        raise ValueError(f"search_mode must be one of {', '.join(MODES)}")
    if not search_enabled(cell):
        if cell["search_mode"] != "exact":
            raise ValueError(f"search_mode blas belongs to {SEARCH_PROFILE}")
        return
    if cell["stage"] != "store-search" or cell["corpus"] != TRAIN:
        raise ValueError(f"{SEARCH_PROFILE} needs the stage store-search and the corpus {TRAIN}")
    if cell["table_dtype"] != "float32" or cell["output_dtype"] not in (
        "float32",
        "float16",
        "int8",
    ):
        raise ValueError(f"{SEARCH_PROFILE} stores rows of a float32 table")
    if cell["base_doc_ids"] != cell["candidate_doc_ids"] or cell["filter_kind"] == "doc_ids":
        raise ValueError(f"{SEARCH_PROFILE} compares the same store without a document filter")
    if cell["resident_cycles"] != 1 or POOL % cell["queries"]:
        raise ValueError(f"{SEARCH_PROFILE} stores the corpus once and takes blocks that fit 256")


def pool(model, job, store_set):
    """(float32 queries (256, dim), receipt): STS-B test texts outside the store, valid
    encodings only, PCG64 seed 0; the same rule as the prefilter count (DR-168 gate)."""
    import numpy as np
    from _development_process import checked
    from _harness.records import document
    from _quality.prefilter_queries import SETS, _candidates, _valid

    corpus, select = SETS["stsb-test"]
    record = job["query_corpora"][corpus]
    texts, info = _candidates(
        select(document(checked(record["path"], record["sha256"]))), store_set
    )
    rows, good = _valid(model, texts)
    valid = [t for t, ok in zip(texts, good, strict=True) if ok]
    if len(valid) < POOL:
        raise ValueError(f"STS-B test has {len(valid)} usable queries, {POOL} needed")
    pick = np.sort(np.random.Generator(np.random.PCG64(0)).choice(len(valid), POOL, False))
    if store_set.intersection(valid[i] for i in pick):
        raise ValueError("the queries share a text with the stored corpus")
    queries = np.ascontiguousarray(rows[good][pick], dtype=np.float32)
    queries.flags.writeable = False
    return queries, {**info, "selected": POOL, "disjoint_checked": True}


class Rotation:
    """The next block of queries per call; ``restart`` begins a group at the first block."""

    def __init__(self, queries, width):
        self.queries, self.width, self.next = queries, width, 0

    def restart(self):
        self.next = 0

    def block(self):
        begin = self.next * self.width % len(self.queries)
        self.next += 1
        return self.queries[begin : begin + self.width]


def metrics(job):
    """What the record of an arm says about the workload of the profile."""
    if not search_enabled(job):
        return {}
    return {
        "search_mode": job["search_mode"],
        "query_pool": job.get("query_receipt"),
        "query_digest": job.get("query_digest"),
        "store_duplicate_text_share": job.get("duplicate_text_share"),
    }
