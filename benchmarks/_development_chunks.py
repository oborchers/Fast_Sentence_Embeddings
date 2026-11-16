"""Private pipeline chunk control, with a public encode oracle outside the timer."""

from __future__ import annotations


def validate(cell):
    size = cell["chunk_bytes"]
    if size is None:
        return
    if type(size) is not int or size < 0:
        raise ValueError("chunk_bytes requires a nonnegative integer or null")
    if cell["stage"] != "encode-memory" or cell["workers"] != 1:
        raise ValueError("chunk control requires one-worker encode-memory")
    if cell["candidate_lengths"] != "none":
        raise ValueError("chunk control does not support length reports")


def operation(model, texts, job):
    import numpy as np
    from _development_encode import output_identity
    from _development_memory import _Target

    from fse._text import StringSource

    dtype = job["output_dtype"]
    codes = np.empty((len(texts), model.output_dim), dtype=dtype)
    scales = np.empty(len(texts), dtype=np.float32) if dtype == "int8" else None
    expected = output_identity(model.encode(texts, workers=1, dtype=dtype))
    pipeline = model._pipeline("cpu")
    size = job["chunk_bytes"] or None  # zero explicitly selects the automatic plan

    def encode():
        sink = _Target(codes, scales, dtype)
        pipeline.infer(StringSource(texts), workers=1, sink=sink, chunk_bytes=size)
        return sink.result()

    if output_identity(encode()) != expected:
        raise ValueError("chunk-controlled pipeline differs from public encode")
    source = StringSource(texts)
    row_bytes = pipeline._row_bytes(dtype, False)
    plan, window, _ = pipeline.plan_tasks(source, 1, size, row_bytes)
    tasks = list(plan.tasks)
    # Obtain actual encoded input bytes by row, independent of scan's internal offsets.
    byte_lengths = [len(text.encode("utf-8")) for text in texts]
    sizes = [sum(byte_lengths[t.first_row : t.first_row + t.n_rows]) for t in tasks]
    rows = [t.n_rows for t in tasks]
    encode.chunk_evidence = dict(
        requested_chunk_bytes=size,
        tasks=len(tasks),
        window=window,
        inline=plan.inline,
        input_bytes=sum(byte_lengths),
        chunk_bytes_min=min(sizes),
        chunk_bytes_max=max(sizes),
        chunk_rows_min=min(rows),
        chunk_rows_max=max(rows),
        output_matches_public_encode=True,
        path="Pipeline.infer, fresh StringSource and Sink per operation",
    )
    return encode
