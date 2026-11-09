"""Untimed search stores and independent per-query binary references."""

from __future__ import annotations

from pathlib import Path

from _harness.records import append, sha256
from _speed_search_identity import require_bits
from _speed_search_sources import queries, sources


def admission(context, phase, estimate):
    admitted = context["ledger"].admit(estimate)
    append(
        context["ledger"].root / "search-admission.jsonl",
        dict(phase=phase, estimated_seconds=estimate, admitted=admitted),
    )
    if not admitted:
        raise TimeoutError(f"search {phase}: budget or stop signal exhausted")


def prefilter(store, cell):
    from fse import _native
    from fse._backends import _cpu

    if not hasattr(_cpu, "_PREFILTER_ENABLED"):
        raise ValueError("search adapter requires the explicit prefilter enable switch")
    if cell["prefilter"] == "off":
        _cpu._PREFILTER_ENABLED = frozenset()
        return False
    if store._backend.prefilter_variant() is not None:
        return False
    if not cell.get("functional_only"):
        raise ValueError("publication prefilter cell requires a shipped enabled variant")
    variant = _cpu._PREFILTER_VARIANT.get(store._backend.isa)
    if variant and _native.prefilter_can_run(variant):
        _cpu._PREFILTER_ENABLED = frozenset({variant})
        return True
    return False


def save_array(path, array):
    import numpy as np

    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    return str(path)


def search_pool(store, pool, cell, candidates=None):
    """Results retained in call order, including the final measured pass."""
    output = []
    q = cell["queries_per_call"]
    for start in range(0, len(pool), q):
        kwargs = {"rows": candidates[start]} if candidates is not None else {}
        if candidates is not None and q != 1:
            raise ValueError("per-query binary reference requires queries_per_call=1")
        result = store.search(
            pool[start : start + q],
            k=cell["k"],
            mode=cell["mode"],
            workers=cell["workers"],
            **kwargs,
            **({"oversample": cell["oversample"]} if cell["mode"] == "binary" else {}),
        )
        output.append(result)
    return output


def arrays(results):
    import numpy as np

    return {
        "ids": np.concatenate([r.ids for r in results]),
        "scores": np.concatenate([r.scores for r in results]),
    }


def candidates(store, pool, cell):
    import numpy as np

    from fse import _reference_bits as bits

    require_bits(store)
    width = min(cell["texts"], cell["oversample"] * cell["k"])
    keep = [np.empty(0, np.int64) for _ in pool]
    query_bits = bits.query_bits(pool)
    # Independently derive bits from stored vectors and rank by (distance, physical row).
    for start in range(0, cell["texts"], 4096):
        vectors = store.get_vectors(start=start, stop=min(cell["texts"], start + 4096))
        distances = bits.hamming(bits.sign_bits(vectors), query_bits)
        positions = np.arange(start, start + len(vectors), dtype=np.int64)
        for index in range(len(pool)):
            keys = (distances[:, index].astype(np.int64) << 41) | positions
            merged = np.concatenate((keep[index], keys))
            if len(merged) > width:
                merged.partition(width - 1)
            keep[index] = merged[:width].copy()
    return np.asarray([np.sort(keys & ((1 << 41) - 1)) for keys in keep])


def references(context, cells, stores, pool, root):
    import numpy as np

    from fse import VectorStore

    admission(context, "references", 120 * len(cells))
    result = {}
    for dtype, path in stores.items():
        cell = dict(cells[0], store_dtype=dtype, prefilter="off", mode="exact")
        with VectorStore.open(path) as store:
            prefilter(store, cell)
            for key, array in arrays(search_pool(store, pool, cell)).items():
                result[f"{dtype}-{key}"] = save_array(root / f"{dtype}-{key}.npy", array)
            if dtype == "float32" and any(c["mode"] == "binary" for c in cells):
                binary = next(c for c in cells if c["mode"] == "binary")
                selected = candidates(store, pool, binary)
                result["candidates"] = save_array(root / "candidates.npy", selected)
                expected = arrays(search_pool(store, pool, cell, selected))
                for key, array in expected.items():
                    result[f"binary-{key}"] = save_array(root / f"binary-{key}.npy", array)
    if not np.isfinite(pool).all():
        raise ValueError("search queries became nonfinite")
    return result


def prepare(context, cells, root):
    import numpy as np

    from fse import Encoder, ShardSink

    admission(context, "model preparation", 300)
    cell = cells[0]
    model = context["models"][cell["model"]]
    encoder = Encoder.from_pretrained(model["artifact"], mmap=True)
    train, test, metadata = sources(context, cell)
    query_file = root / "queries.npy"
    query_metadata = queries(encoder, test, cell, query_file)
    del test
    stores = {}
    for dtype in sorted({c["store_dtype"] for c in cells}):
        admission(context, f"{dtype} store build", 180)
        path = root / dtype
        with ShardSink(
            path,
            encoder,
            dtype=dtype,
            sign_bits=dtype == "float32",
            shard_rows=16384,
            durability="none",
        ) as sink:
            encoder.encode(train, sink=sink, workers=cell["workers"])
        stores[dtype] = str(path)
    del encoder, train
    pool = np.load(query_file, allow_pickle=False)
    reference_files = references(context, cells, stores, pool, root)
    append(
        context["ledger"].root / "inputs.jsonl",
        dict(
            model=cell["model"],
            search_sources=metadata,
            search_queries=query_metadata,
            reference_digests={key: sha256(Path(p)) for key, p in reference_files.items()},
        ),
    )
    return stores, str(query_file), reference_files
