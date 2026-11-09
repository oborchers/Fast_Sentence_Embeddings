"""Pinned train paragraphs and disjoint, unique test queries for search."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from _harness import functional
from _harness.records import append, atomic, sha256
from _speed_cells import parquet_texts


def texts(folder, configuration, split):
    import pyarrow.parquet as pq

    parts = sorted((folder / configuration).glob(f"{split}-*.parquet"))
    if not parts:
        raise ValueError(f"missing pinned search split {split}")
    return (text for part in parts for text in parquet_texts(pq.ParquetFile(part)))


def sources(context, cell):
    from huggingface_hub import snapshot_download

    contract = context["ledger"].contract
    spec = contract["corpus"]
    folder = Path(
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            repo_type="dataset",
            cache_dir=context["datasets"],
            allow_patterns=[
                f"{spec['configuration']}/{split}-*.parquet" for split in ("train", "test")
            ],
        )
    )
    count = spec["expected_paragraphs"]
    positions = functional.select(count, cell["texts"])
    if len(positions) != cell["texts"]:
        raise ValueError("search store size exceeds the pinned source")
    hashes, selected_bytes, full_bytes = inventory(folder, spec, set(positions))
    if full_bytes > contract["search_queries"]["source_utf8_bound_bytes"]:
        raise ValueError("search source exceeds the declared memory bound")
    candidates, counts = test_candidates(folder, spec, hashes)
    context["search_selected_bytes"] = selected_bytes
    from _speed_search_plan import admit

    admit(context, context["search_group_cells"])
    wanted = set(positions)
    train = [
        text for i, text in enumerate(texts(folder, spec["configuration"], "train")) if i in wanted
    ]
    expected = contract["search_queries"]["expected_test_sentences"]
    if expected is None and not cell.get("functional_only"):
        raise ValueError(
            "pin search test sentence count from the functional probe before publication"
        )
    if expected is not None and counts["test_sentences"] != expected:
        raise ValueError("pinned search test sentence count changed")
    metadata = {
        **counts,
        "eligible_texts": len(candidates),
        "train_paragraphs": count,
        "train_utf8_bytes": full_bytes,
        "duplicate_store_texts": count - len(hashes),
        "duplicate_store_fraction": 1 - len(hashes) / count,
        "corpus": spec,
        "query_split": "test",
        "functional": functional.reduction(count, positions, floor=65536)
        if cell.get("functional_only")
        else None,
    }
    append(context["ledger"].root / "search-sources.jsonl", dict(cell_id=cell["id"], **metadata))
    return train, candidates, metadata


def inventory(folder, spec, wanted):
    hashes, selected, total, count = set(), 0, 0, 0
    for index, text in enumerate(texts(folder, spec["configuration"], "train")):
        count += 1
        size = len(text.encode()) + 1
        total += size
        hashes.add(hashlib.sha256(text.encode()).digest())
        if index in wanted:
            selected += size
    if count != spec["expected_paragraphs"]:
        raise ValueError("pinned search train paragraph count changed")
    return hashes, selected, total


def test_candidates(folder, spec, hashes):
    candidates, seen, counts = [], set(), dict(test_sentences=0, duplicates=0, in_store=0)
    size = 0
    for paragraph in texts(folder, spec["configuration"], "test"):
        for text in re.split(r"(?<=[.!?])\s+", paragraph):
            if len(text.split()) < 5:
                continue
            counts["test_sentences"] += 1
            digest = hashlib.sha256(text.encode()).digest()
            if digest in hashes:
                counts["in_store"] += 1
            elif digest in seen:
                counts["duplicates"] += 1
            else:
                size += len(text.encode()) * 4 + 200
                if size > 32 * 1024**2:
                    raise MemoryError("search test candidates exceed their memory bound")
                candidates.append(text)
                seen.add(digest)
    return candidates, counts


def queries(encoder, candidates, cell, path):
    import numpy as np

    rng = np.random.Generator(np.random.PCG64(0))
    order = rng.permutation(len(candidates))
    vectors, selected, rejected, examined = [], [], 0, 0
    for start in range(0, len(order), 256):
        positions = order[start : start + 256]
        batch = encoder.encode([candidates[int(i)] for i in positions], workers=cell["workers"])
        for position, vector in zip(positions, batch, strict=True):
            examined += 1
            if not np.isfinite(vector).all() or not np.any(vector.view(np.uint32) & 0x7FFFFFFF):
                rejected += 1
                continue
            vectors.append(vector.copy())
            selected.append(candidates[int(position)])
            if len(vectors) == cell["pool_size"]:
                with path.open("xb") as stream:
                    np.save(stream, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
                metadata = {
                    "sha256": sha256(path),
                    "pool_size": len(vectors),
                    "seed": 0,
                    "sampling": "PCG64 permutation without replacement, fixed order",
                    "examined": examined,
                    "nonfinite_or_zero": rejected,
                    "remaining": len(candidates) - examined,
                    "texts_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
                }
                atomic(path.with_suffix(".json"), metadata)
                return metadata
    raise ValueError("not enough unique, disjoint, finite nonzero search queries")
