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

    observed = []

    class ObservedSource(StringSource):
        def encode(self, chunk, lookup, workspace):
            observed.append(
                dict(
                    first_row=chunk.first_row, rows=chunk.n_sentences, bytes=self._span_bytes(chunk)
                )
            )
            return super().encode(chunk, lookup, workspace)

    # Record only this real verification run. The timed closure has no observer.
    source = ObservedSource(texts)
    sink = _Target(codes, scales, dtype)
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
        observation="source.encode calls in the untimed verification run",
        output_matches_public_encode=True,
        path="Pipeline.infer, fresh StringSource and Sink per operation",
    )
    return encode
