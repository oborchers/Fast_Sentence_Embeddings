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

    from fse._text import Chunk, StringSource

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

    observed = []

    source = StringSource(texts)

    class ObservedTarget(_Target):
        def target(self, first_row, n_rows):
            chunk = Chunk(first_row, first_row + n_rows, first_row, n_rows)
            observed.append(dict(first_row=first_row, rows=n_rows, bytes=source._span_bytes(chunk)))
            return super().target(first_row, n_rows)

    # The executor reserves one target per actual task. Only this verification
    # sink records the ranges, including the fused text/pooling worker path.
    sink = ObservedTarget(codes, scales, dtype)
    pipeline.infer(source, workers=1, sink=sink, chunk_bytes=size)
    if output_identity(sink.result()) != expected:
        raise ValueError("chunk-controlled pipeline differs from public encode")
    if output_identity(encode()) != expected:
        raise ValueError("unobserved chunk-controlled pipeline differs from public encode")
    row_bytes = pipeline._row_bytes(dtype, False)
    window, row_limit = pipeline._window(1, row_bytes)

    if sum(task["rows"] for task in observed) != len(texts):
        raise ValueError("observed tasks do not cover the verification input")
    if sum(task["bytes"] for task in observed) != source.nbytes:
        raise ValueError("observed task bytes differ from source storage")
    if any(task["rows"] > row_limit for task in observed):
        raise ValueError("observed task exceeds the pipeline row limit")

    def quantiles(name):
        values = [task[name] for task in observed]
        return np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist() if values else None

    encode.chunk_evidence = dict(
        requested_chunk_bytes=size,
        tasks=len(observed),
        window=window,
        row_limit=row_limit,
        input_bytes=source.nbytes,
        byte_quantiles=quantiles("bytes"),
        row_quantiles=quantiles("rows"),
        quantile_probabilities=[0, 0.25, 0.5, 0.75, 1],
        observed_tasks=observed,
        observation="Sink.target reservations in the untimed verification run",
        output_matches_public_encode=True,
        path="Pipeline.infer, fresh StringSource and Sink per operation",
    )
    return encode
