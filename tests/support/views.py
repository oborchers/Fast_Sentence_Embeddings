"""Stores written in parts and in one piece, for the tests of views."""

import numpy as np

import fse
from support import scores as sc
from support.encoder import finished
from support.runtime import texts

SIZES = (13, 9, 18)


def sentences(n):
    """Sentences with repeats: the rows of a repeated sentence have equal scores."""
    base = texts(max(1, n // 2))
    return [base[i % len(base)] for i in range(n)]


def write_vectors(path, source, ids=None, *, dtype="float32", shard_rows=7):
    """One store of ``source`` (and document IDs ``ids``) from a real run."""
    with fse.ShardSink(
        path, finished(dim=17), dtype=dtype, shard_rows=shard_rows, doc_ids=ids is not None
    ) as sink:
        finished(dim=17).encode(source, doc_ids=ids, sink=sink, workers=1)


def vector_parts(tmp_path, sizes=SIZES, *, doc_ids=True, dtype="float32"):
    """The whole store and the paths of stores that hold its rows in consecutive parts,
    each with another shard size."""
    n = sum(sizes)
    source = sentences(n)
    ids = np.arange(n, dtype=np.int64)[::-1].copy() % 7 if doc_ids else None
    write_vectors(tmp_path / "whole", source, ids, dtype=dtype)
    paths, first = [], 0
    for k, rows in enumerate(sizes):
        part = ids[first : first + rows] if doc_ids else None
        write_vectors(
            tmp_path / f"part{k}", source[first : first + rows], part, dtype=dtype, shard_rows=3 + k
        )
        paths.append(tmp_path / f"part{k}")
        first += rows
    return tmp_path / "whole", paths


def score_values(n, dim, seed=3):
    """Distinct scores with some equal rows."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, dim)).astype(np.float32)
    values[n // 2 :: 3] = values[0]
    return values


def score_parts(tmp_path, sizes=SIZES, *, doc_ids=False, limits=None):
    """The whole score store and the paths of stores that hold its rows in consecutive
    parts. ``limits`` gives the threshold of each part (default 0.5)."""
    info = sc.default_info(doc_ids)
    values = score_values(sum(sizes), info.dim)
    sc.make_scores(tmp_path / "whole", info, sizes=())
    sc.add_shard(tmp_path / "whole", info, 0, len(values), scores=values)
    paths, first = [], 0
    for k, rows in enumerate(sizes):
        path = tmp_path / f"part{k}"
        sc.make_scores(path, info, sizes=())
        limit = 0.5 if limits is None else limits[k]
        sc.add_shard(
            path, info, 0, rows // 2, scores=values[first : first + rows // 2], limit=limit
        )
        sc.add_shard(
            path,
            info,
            rows // 2,
            rows - rows // 2,
            scores=values[first + rows // 2 : first + rows],
            limit=limit,
        )
        paths.append(path)
        first += rows
    return tmp_path / "whole", paths, values
