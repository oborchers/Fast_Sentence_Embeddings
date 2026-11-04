"""Full direct/store output exports and exact STS scores, never throughput evidence."""

from __future__ import annotations

import gc
import math
import tempfile
from pathlib import Path

from _development_budget import check
from _development_process import checked
from _measurement import document, sha256


def _save(root, name, array):
    import numpy as np

    path = root / (name + ".npy")
    if path.exists():
        raise FileExistsError(path)
    np.save(path, array, allow_pickle=False)
    return _record(path, array)


def _record(path, array):
    return dict(path=str(path), sha256=sha256(path), shape=list(array.shape), dtype=array.dtype.str)


def _scores(vectors, sections):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    result = {}
    for section in sections:
        begin, pairs = section["begin"], section["pairs"]
        rows = vectors[begin : begin + 2 * pairs]
        labels = np.asarray(section["labels"], dtype=np.float64)
        if len(rows) != 2 * pairs or len(labels) != pairs:
            raise ValueError("incomplete STS-B split")
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        normalized = np.divide(rows, norms, out=np.zeros_like(rows), where=norms != 0)
        similarities = (normalized[:pairs] * normalized[pairs:]).sum(axis=1)
        values = {
            "pearson": float(pearsonr(labels, similarities).statistic * 100),
            "spearman": float(spearmanr(labels, similarities).statistic * 100),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("nonfinite STS-B score")
        result[section["split"]] = {"pairs": pairs, **values}
    return result


def _direct(model, texts, root, workers, dtype):
    from fse._backends._cpu import CpuBackend

    rows = model.encode(texts, dtype=dtype, workers=workers)
    codes = rows.codes if dtype == "int8" else rows
    tensors = {"vectors": _save(root, dtype + "-direct-vectors", codes)}
    if dtype == "int8":
        tensors["scales"] = _save(root, dtype + "-direct-scales", rows.scales)
    inverse = CpuBackend().inverse_norms(codes)
    tensors["inv_norms"] = _save(root, dtype + "-direct-inv_norms", inverse)
    return tensors, rows


def _stored(job, model, texts, root, dtype):
    import numpy as np

    from fse import ShardSink, VectorStore

    arrays = {}
    with tempfile.TemporaryDirectory(prefix="quality-store-", dir=job["work"]) as temporary:
        path = Path(temporary) / "store"
        with ShardSink(path, model, dtype=dtype, durability="none") as sink:
            model.encode(texts, workers=job["workers"], sink=sink)
        with VectorStore.open(path) as store:
            store.verify()
            if len(store) != len(texts):
                raise ValueError("quality store row count differs")
            for shard, tensors in store._shards:
                _copy_shard(root, dtype, len(texts), shard, tensors, arrays)
        for array in arrays.values():
            array.flush()
    result = {
        name: _record(root / (dtype + "-store-" + name + ".npy"), array)
        for name, array in arrays.items()
    }
    expected = {"vectors", "inv_norms"} | ({"scales"} if dtype == "int8" else set())
    if set(result) != expected:
        raise ValueError("quality store tensor inventory differs")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("quality store contains nonfinite output")
    return result


def _copy_shard(root, dtype, count, shard, tensors, arrays):
    import numpy as np

    for name in ("vectors", "scales", "inv_norms"):
        if name not in tensors:
            continue
        source = tensors[name]
        if name not in arrays:
            path = root / (dtype + "-store-" + name + ".npy")
            arrays[name] = np.lib.format.open_memmap(
                path, mode="w+", dtype=source.dtype, shape=(count, *source.shape[1:])
            )
        arrays[name][shard.first_row : shard.first_row + shard.rows] = source


def export(job):
    from fse import Encoder

    artifact = job["artifact"]
    model = Encoder.load(checked(artifact["path"], artifact["sha256"]), mmap=True)
    corpus = job["corpus"]
    data = document(checked(corpus["path"], corpus["sha256"]))
    texts = data["texts"]
    if len(texts) != corpus["texts"]:
        raise ValueError("quality input row count differs")
    root = Path(job["output"])
    root.mkdir(parents=True, exist_ok=False)
    outputs, scores = {}, {}
    for dtype in ("float32", "float16", "int8"):
        check(job.get("wall_deadline"))
        direct, rows = _direct(model, texts, root, job["workers"], dtype)
        if dtype == "float32":
            scores = _scores(rows, data["score_slices"])
        del rows
        gc.collect()
        check(job.get("wall_deadline"))
        stored = _stored(job, model, texts, root, dtype)
        outputs[dtype] = {"direct": direct, "stored": stored}
    return {
        "outputs": outputs,
        "scores": scores,
        "encoder_fingerprint": model.fingerprint,
        "score_definition": "original complete STS-B splits; float32 cosine, float64 Pearson/Spearman times 100",
    }
