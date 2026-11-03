"""Complete local encoding outputs and STS scores, never throughput measurements."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from _measurement import atomic, document, package_identity, sha256
from _speed_cells import configure


def _save(directory, name, array):
    import numpy as np

    path = directory / (name + ".npy")
    np.save(path, array, allow_pickle=False)
    return {
        "file": str(path),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": sha256(path),
    }


def _scores(vectors, sections):
    import numpy as np
    from hub_classic_metrics import normalized, score
    from scipy.stats import spearmanr

    result = {}
    for section in sections:
        begin, pairs = section["begin"], section["pairs"]
        rows = vectors[begin : begin + 2 * pairs]
        labels = np.asarray(section["labels"], dtype="float64")
        if len(rows) != 2 * pairs or len(labels) != pairs:
            raise ValueError("STS score section length differs from its complete split")
        rows = normalized(rows)
        similarities = (rows[:pairs] * rows[pairs:]).sum(axis=1)
        values = {
            "pearson": score(labels, vectors[begin : begin + 2 * pairs]),
            "spearman": float(spearmanr(labels, similarities).statistic * 100),
        }
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("STS score is not finite")
        result[section["split"]] = {**values, "pairs": pairs}
    return result


def _export_store(store, texts, directory, ids, dtype):
    import numpy as np

    result = {}
    store.verify()
    if len(store) != len(texts):
        raise ValueError("stored quality output has the wrong row count")
    for shard, tensors in store._shards:
        if ids is not None and not np.array_equal(
            tensors["row_ids"], ids[shard.first_row : shard.first_row + shard.rows]
        ):
            raise ValueError("quality output document IDs are misaligned")
        for name in ("vectors", "scales", "inv_norms"):
            if name in tensors:
                key = f"{dtype}-{shard.first_row}-{name}"
                result[key] = _save(directory, key, tensors[name])
    return result


def _stored(job, model, texts, directory, ids):
    from fse import ShardSink, VectorStore

    result = {}
    for dtype in ("float32", "int8"):
        with tempfile.TemporaryDirectory(dir=job["work"]) as temporary:
            path = Path(temporary) / "store"
            options = {} if ids is None else {"doc_ids": True}
            with ShardSink(path, model, dtype=dtype, **options) as sink:
                model.encode(
                    texts,
                    sink=sink,
                    workers=job["workers"],
                    **({} if ids is None else {"doc_ids": ids}),
                )
            with VectorStore.open(path) as store:
                result.update(_export_store(store, texts, directory, ids, dtype))
    return result


def child(job):
    configure(job["workers"])
    import numpy as np

    from fse import Encoder

    identity = package_identity("fse")
    for key in ("input", "artifact"):
        if sha256(Path(job[key])) != job[key + "_sha256"]:
            raise ValueError(f"quality {key} changed")
    data = document(Path(job["input"]))
    texts = data["texts"]
    if len(texts) != job["texts"]:
        raise ValueError("quality input row count changed")
    model = Encoder.load(job["artifact"], mmap=True)
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=False)
    direct = {}
    vectors = model.encode(texts, workers=job["workers"])
    direct["float32"] = _save(output, "encoded-float32", vectors)
    scores = _scores(vectors, data.get("score_slices", []))
    del vectors
    quantized = model.encode(texts, dtype="int8", workers=job["workers"])
    direct["int8-codes"] = _save(output, "encoded-int8-codes", quantized.codes)
    direct["int8-scales"] = _save(output, "encoded-int8-scales", quantized.scales)
    del quantized
    modes = {"plain": direct}
    for enabled in [False, True] if job.get("doc_ids") else [False]:
        directory = output / ("ids" if enabled else "no-ids")
        directory.mkdir()
        ids = np.arange(len(texts), dtype="int64") // 4 if enabled else None
        modes["ids" if enabled else "no-ids"] = _stored(job, model, texts, directory, ids)
    if package_identity("fse") != identity:
        raise ValueError("executing fse changed during quality evaluation")
    atomic(
        Path(job["receipt"]),
        {
            "identity": identity,
            "outputs": modes,
            "scores": scores,
            "encoder_fingerprint": model.fingerprint,
            "texts": len(texts),
            "local_diagnostic": True,
            "publishable": False,
            "score_definition": "Pearson and Spearman multiplied by 100, saved default tokenizer",
        },
    )
