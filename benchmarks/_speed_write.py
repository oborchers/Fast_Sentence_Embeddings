"""Private helpers for the central measurement entries."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from _harness.records import atomic, sha256
from _measurement import child_record, package_identity, telemetry
from _speed_cells import configure


def headline_tokens(snapshot, source):
    import numpy as np

    from fse import StaticModel

    model = StaticModel.from_pretrained(snapshot)
    if model.max_length != 512:
        raise ValueError("headline requires the pinned 512-token policy")
    counts = np.empty(len(source), dtype=np.int64)
    for start in range(0, len(source), 2048):
        batch = [
            text[: model.max_length * model.median_token_length]
            for text in source[start : start + 2048]
        ]
        rows = model.tokenizer.encode_batch(batch, add_special_tokens=False)
        counts[start : start + len(rows)] = [len(row.ids[: model.max_length]) for row in rows]
    return counts


def headline_child(job):
    import resource

    import numpy as np

    from fse import Encoder, ShardSink, VectorStore

    configure(job["workers"])
    model = Encoder.load(job["artifact"], mmap=True)
    path = Path(job["vectors_file"])
    if path.exists() or path.is_symlink():
        raise ValueError("headline output must be fresh")
    if sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("headline input changed")
    identity = package_identity("fse")
    before = telemetry()
    disk_before = __import__("shutil").disk_usage(path.parent)._asdict()
    started = time.perf_counter()
    sink = ShardSink(path, model, dtype="int8", durability="none")
    first_read = time.perf_counter()
    with sink:
        model.encode(Path(job["input_file"]), workers=job["workers"], sink=sink)
    seconds = time.perf_counter() - first_read
    including_open = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    after = telemetry()
    if sink.published_rows != job["texts"]:
        raise ValueError("headline row count differs")
    files, synced_at, sync_telemetry = sync_headline(path)
    durable_seconds = synced_at - first_read
    indices = json.loads(Path(job["sample_file"]).read_text())
    sample_path = Path(job["sample_texts_file"])
    if sha256(sample_path) != job["sample_texts_sha256"]:
        raise ValueError("headline sample texts changed")
    texts = __import__("json").loads(sample_path.read_text())
    reference = model.encode(texts, dtype="int8", workers=job["workers"])
    with VectorStore.open(path, encoder=model) as store:
        if len(store) != job["texts"]:
            raise ValueError("reopened headline row count differs")
        store.verify()
        codes, scales = stored_codes(store, indices)
    if not np.array_equal(reference.codes.view(np.uint8), codes.view(np.uint8)):
        raise ValueError("headline int8 sample codes differ")
    if not np.array_equal(reference.scales.view(np.uint8), scales.view(np.uint8)):
        raise ValueError("headline int8 sample scales differ")
    if package_identity("fse") != identity:
        raise ValueError("executing headline backend changed")
    atomic(
        Path(job["receipt"]),
        dict(
            **child_record(job),
            seconds=seconds,
            affinity=sorted(os.sched_getaffinity(0)),
            requested_worker_limit=job["workers"],
            including_sink_open_seconds=including_open,
            time_until_durable_seconds=durable_seconds,
            external_fsync=True,
            durability="none",
            published_rows=sink.published_rows,
            durable_rows=sink.durable_rows,
            output_dtype="int8",
            table_dtype=str(model.word_vectors.dtype),
            sample_checked_rows=len(indices),
            sample_bitwise_equal=True,
            peak_rss_bytes=peak,
            identity=identity,
            files=files,
            bytes_written=sum(row["bytes"] for row in files),
            telemetry_before=before,
            telemetry_after=after,
            telemetry_after_sync=sync_telemetry,
            disk_before=disk_before,
            disk_with_output=__import__("shutil").disk_usage(path.parent)._asdict(),
        ),
    )


def sync_headline(path):
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_symlink():
            raise ValueError("unexpected symlink in owned headline output")
        if p.is_file():
            with p.open("rb") as stream:
                os.fsync(stream.fileno())
            files.append(p)
    for p in [*sorted((p for p in path.rglob("*") if p.is_dir()), reverse=True), path, path.parent]:
        descriptor = os.open(p, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    synced_at, sync_telemetry = time.perf_counter(), telemetry()
    inventory = [
        dict(path=str(p.relative_to(path)), bytes=p.stat().st_size, sha256=sha256(p)) for p in files
    ]
    return inventory, synced_at, sync_telemetry


def stored_codes(store, indices):
    import numpy as np

    indices = np.asarray(indices, dtype=np.int64)
    codes = np.empty((len(indices), store.dim), dtype=np.int8)
    scales = np.empty(len(indices), dtype=np.float32)
    for shard, tensors in store._shards:
        mask = (indices >= shard.first_row) & (indices < shard.first_row + shard.rows)
        rows = indices[mask] - shard.first_row
        codes[mask] = tensors["vectors"][rows]
        scales[mask] = tensors["scales"][rows]
    return codes, scales
