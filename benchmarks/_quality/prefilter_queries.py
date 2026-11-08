"""The two stand-in query sets of the prefilter count, disjoint from the stored texts."""

from __future__ import annotations

import numpy as np
from _development_process import checked
from _harness.records import document


def _stsb_test(data):
    section = next(s for s in data["score_slices"] if s["split"] == "test")
    return data["texts"][section["begin"] : section["begin"] + 2 * section["pairs"]]


def _wikitext_test(data):
    rows = (" ".join(text.split()) for text in data["texts"])
    return [t for t in rows if t and not t.startswith("=") and len(t.split()) >= 5]


SETS = {"stsb-test": ("stsb", _stsb_test), "wikitext-test": ("wikitext", _wikitext_test)}


def _candidates(texts, store_set):
    """Normalized, unique texts that are not stored; the exclusions are counted."""
    seen, kept = set(), []
    for text in (" ".join(t.split()) for t in texts):
        if text and text not in seen:
            seen.add(text)
            kept.append(text)
    free = [t for t in kept if t not in store_set]
    return free, {
        "input": len(texts),
        "unique": len(kept),
        "in_store_excluded": len(kept) - len(free),
    }


def _valid(model, texts):
    rows = model.encode(texts, workers=1)
    good = np.isfinite(rows).all(axis=1) & (rows != 0).any(axis=1)
    return rows, good


def query_sets(model, job, store_set):
    """name -> (float32 queries, receipt): PCG64 seed 0 over the valid candidates."""
    result = {}
    for name, (corpus, select) in SETS.items():
        record = job["query_corpora"][corpus]
        data = document(checked(record["path"], record["sha256"]))
        texts, info = _candidates(select(data), store_set)
        rows, good = _valid(model, texts)
        info["invalid_encoding_excluded"] = int((~good).sum())
        valid = [t for t, ok in zip(texts, good, strict=True) if ok]
        count = job["queries_per_set"]
        if len(valid) < count:
            raise ValueError(f"{name} has {len(valid)} usable queries, {count} needed")
        pick = np.sort(np.random.Generator(np.random.PCG64(0)).choice(len(valid), count, False))
        chosen = [valid[i] for i in pick]
        if store_set.intersection(chosen):
            raise ValueError(f"{name} shares a text with the stored corpus")
        queries = np.ascontiguousarray(rows[good][pick], dtype=np.float32)
        result[name] = (queries, {**info, "selected": count, "disjoint_checked": True})
    return result
