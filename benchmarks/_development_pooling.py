"""Controlled pooling layouts for local working-set attribution, outside every timer."""

from __future__ import annotations

from dataclasses import replace

LAYOUTS = ("original", "footprint-wide", "footprint-compact")


def validate(cell):
    for name in ("base_pooling_layout", "candidate_pooling_layout"):
        if cell[name] not in LAYOUTS:
            raise ValueError("unknown pooling layout")
        if cell[name] != "original" and (
            cell["stage"] != "pooling" or cell["table_dtype"] != "float32"
        ):
            raise ValueError("footprint layouts require float32 pooling")


def configure(job, cell, name, control):
    arm = "base" if control or name == "base" else "candidate"
    job["pooling_layout"] = cell[f"{arm}_pooling_layout"]


def _aligned_copy(values):
    import numpy as np

    storage = np.empty(values.nbytes + 127, dtype=np.uint8)
    offset = -storage.ctypes.data % 128
    result = np.ndarray(values.shape, dtype=values.dtype, buffer=storage, offset=offset)
    result[...] = values
    return result


def prepare(model, batch, job):
    """Repeat the same numeric rows in a wide table, or remap to its compact source.

    Both arms contribute identical values and weights in identical order. Dimension,
    row stride, tile capacity, output size and prefetch instruction count stay fixed.
    This is a synthetic locality control, not the unchanged model's workload.
    """
    import numpy as np

    from fse._contract import OOV_TOKEN

    pool = model._spec.pool_input()
    layout = job.get("pooling_layout", "original")
    if layout == "original":
        return pool, batch
    if pool.dtype != "float32" or pool.fasttext or job["stage"] != "pooling":
        raise ValueError("footprint control requires float32 pooling without FastText")
    rows = len(pool.vectors)
    compact_rows = max(1, rows // 4)
    source = np.arange(rows, dtype=np.uint32) % compact_rows
    valid = batch.tokens != OOV_TOKEN
    tokens = batch.tokens.copy()
    if layout == "footprint-wide":
        vectors = _aligned_copy(pool.vectors[source])
        weights = _aligned_copy(pool.weights[source])
    elif layout == "footprint-compact":
        vectors = _aligned_copy(pool.vectors[:compact_rows])
        weights = _aligned_copy(pool.weights[:compact_rows])
        tokens[valid] %= compact_rows
    else:
        raise ValueError("unknown pooling layout")
    return replace(pool, vectors=vectors, weights=weights), replace(batch, tokens=tokens)


def evidence(pool, batch, backend, layout):
    """Observed arrays and ISA; capacity is derived for the default 16 KiB kernel only."""
    import numpy as np

    from fse._container._hash import checksum
    from fse._contract import OOV_TOKEN

    vectors = pool.vectors
    rows = len(np.unique(batch.tokens[batch.tokens != OOV_TOKEN]))
    # Mirrors FSE_TILE_BYTES, FSE_TILE_MAX and FSE_TILE_MIN in _native_kernel_tiles.h.
    capacity = 16384 // int(vectors.strides[0])
    tiled = backend.isa == "neon" and pool.dtype == "float32" and capacity >= 4
    return dict(
        layout=layout,
        isa=backend.isa,
        dimension=pool.dim,
        table_rows=len(vectors),
        table_bytes=int(vectors.nbytes),
        row_bytes=int(vectors.strides[0]),
        default_neon_tile_capacity=min(64, capacity) if tiled else None,
        table_alignment_mod128=int(vectors.ctypes.data % 128),
        weights_alignment_mod128=int(pool.weights.ctypes.data % 128),
        distinct_accessed_rows=rows,
        accessed_row_bytes=rows * int(vectors.strides[0]),
        weights_all_one=bool(np.all(pool.weights == np.float32(1))),
        vectors_xxh3=checksum(vectors),
        weights_xxh3=checksum(pool.weights),
        tokens_xxh3=checksum(batch.tokens),
    )
