"""Equal vector and score stores in one pass or two, with untimed proofs and cleanup."""

from __future__ import annotations

import os
import resource
import sys
import time
from contextlib import ExitStack
from importlib.metadata import distributions
from pathlib import Path

from _harness.pressure import interval
from _harness.records import atomic, sha256
from _measurement import child_record, package_identity, telemetry
from _speed_cells import check_output, configure, remove_headline
from _speed_heads import _encoder, _group, sample_indices
from _speed_write import sync_headline


def _predict(heads, job, source, scores, ids, vectors=None):
    return heads.predict(
        source,
        output="decision",
        workers=job["workers"],
        return_vectors=True,
        score_sink=scores,
        vector_sink=vectors,
        doc_ids=ids,
    )


def operation(job, encoder, heads, root):
    """First source read through the final close. Both arms use the unfolded head path."""
    import numpy as np

    from fse import ScoreSink, ShardSink

    source = Path(job["input_file"])
    root.mkdir()
    ids = np.arange(job["texts"], dtype=np.int64)
    calls = []
    with ExitStack() as stack:
        vectors = stack.enter_context(
            ShardSink(root / "vectors", encoder, durability="none", resume=True, doc_ids=True)
        )
        if job["label"] == "fse":
            scores = stack.enter_context(
                ScoreSink(root / "scores", heads, durability="none", resume=True, doc_ids=True)
            )
            first_read = time.perf_counter()
            _predict(heads, job, source, scores, ids, vectors)
            calls.append(time.perf_counter() - first_read)
        elif job["label"] == "fse-two-pass":
            first_read = time.perf_counter()
            encoder.encode(source, workers=job["workers"], sink=vectors, doc_ids=ids)
            calls.append(time.perf_counter() - first_read)
            vectors.close()
            scores = stack.enter_context(
                ScoreSink(root / "scores", heads, durability="none", resume=True, doc_ids=True)
            )
            started = time.perf_counter()
            _predict(heads, job, source, scores, ids)
            calls.append(time.perf_counter() - started)
        else:
            raise ValueError("head sink arm must be fse or fse-two-pass")
    seconds = time.perf_counter() - first_read
    if vectors.published_rows != job["texts"] or scores.published_rows != job["texts"]:
        raise ValueError("head sink published row count differs")
    return {"seconds": seconds, "inference_call_seconds": calls}


def _sample(store, indices, tensor):
    import numpy as np

    width = store.dim if tensor == "vectors" else sum(h.width for h in store.heads.values())
    sample = np.empty((len(indices), width), dtype=np.float32)
    found = np.zeros(len(indices), dtype=bool)
    for shard, tensors in store._shards:
        first, last = shard.first_row, shard.first_row + shard.rows
        ids = tensors["doc_ids"]
        if not np.array_equal(ids, np.arange(first, last, dtype=np.int64)):
            raise ValueError("head sink document IDs differ from row positions")
        mask = (indices >= first) & (indices < last)
        sample[mask] = tensors[tensor][indices[mask] - first]
        found |= mask
    if not found.all():
        raise ValueError("head sink sample is outside the store")
    return sample


def verify(job, root, encoder, heads):
    """Verify both complete stores and their IDs, then sample against the numpy reference."""
    import numpy as np

    from fse import ScoreStore, VectorStore

    indices = sample_indices(job)
    if len(indices) != min(10_000, job["texts"]):
        raise ValueError("head sink sample must have 10,000 rows (all rows in small probes)")
    with VectorStore.open(root / "vectors", encoder=encoder) as store:
        store.verify()
        if len(store) != job["texts"] or not store.has_doc_ids:
            raise ValueError("vector store row count or IDs differ")
        vectors = _sample(store, indices, "vectors")
    with ScoreStore.open(root / "scores", heads=heads) as store:
        store.verify()
        if len(store) != job["texts"] or not store.has_doc_ids:
            raise ValueError("score store row count or IDs differ")
        scores = _sample(store, indices, "scores")
    wanted = set(indices.tolist())
    with Path(job["input_file"]).open() as stream:
        texts = [line.rstrip("\n") for i, line in enumerate(stream) if i in wanted]
    outputs, reference_vectors = heads.predict(
        texts, output="decision", return_vectors=True, backend="numpy"
    )
    check_output(reference_vectors, vectors, exact=True)
    check_output(np.concatenate(list(outputs.values()), axis=1), scores, exact=True)
    return {"indices": indices, "vectors": vectors, "scores": scores}


def synchronized_remove(root):
    """Sync and inventory the one owned output set, then remove it before the next arm."""
    if not root.exists():
        return {"files": [], "logical_file_bytes": 0}
    files, _, after = sync_headline(root)
    remove_headline(root.parent, root, files)
    return {
        "files": files,
        "logical_file_bytes": sum(f["bytes"] for f in files),
        "telemetry_after_sync": after,
        "synchronized_and_removed": True,
    }


def child(job):
    """One full warmup and one timed operation, never retaining two sets of stores."""
    import numpy as np

    from fse import Heads

    configure(job["workers"])
    source, root = Path(job["input_file"]), Path(job["vectors_file"])
    if root.exists() or root.is_symlink():
        raise ValueError("head sink output must be fresh")
    if sha256(source) != job["input_sha256"]:
        raise ValueError("head sink input changed")
    identity = package_identity("fse")
    encoder, heads = _encoder(job), Heads(**_group(job))
    # Initialize both plans before any full warmup or measured input read.
    heads.predict(["warm the unfolded plans"], output="decision", return_vectors=True)
    try:
        warm_before = telemetry()
        operation(job, encoder, heads, root)
        warm_after = telemetry()
        verify(job, root, encoder, heads)
        warmup = {
            **synchronized_remove(root),
            "full_input": True,
            "telemetry_before": warm_before,
            "telemetry_after": warm_after,
            "cpu_pressure": interval(warm_before, warm_after),
        }
        before = telemetry()
        measured = operation(job, encoder, heads, root)
        after = telemetry()
        sample = verify(job, root, encoder, heads)
        sample_path = Path(job["head_sample_file"])
        with sample_path.open("xb") as stream:
            np.savez(stream, **sample)
            stream.flush()
            os.fsync(stream.fileno())
        stores = synchronized_remove(root)
        if sha256(source) != job["input_sha256"] or package_identity("fse") != identity:
            raise ValueError("head sink input or executing backend changed")
        atomic(
            Path(job["receipt"]),
            {
                **child_record(job),
                **measured,
                **stores,
                "warmup": warmup,
                "versions": {d.metadata["Name"]: d.version for d in distributions()},
                "identity": identity,
                "head_sample_file": str(sample_path),
                "head_sample_sha256": sha256(sample_path),
                "sample_checked_rows": len(sample["indices"]),
                "sample_numpy_bitwise_equal": True,
                "durability": "none",
                "external_fsync": True,
                "timer": "first source read through final sink close",
                "workers": job["workers"],
                "worker_path": "native-file-heads-sinks",
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "telemetry_before": before,
                "telemetry_after": after,
                "cpu_pressure": interval(before, after),
            },
        )
    finally:
        synchronized_remove(root)
