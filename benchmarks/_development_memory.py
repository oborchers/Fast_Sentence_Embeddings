"""Repeated complete encodes into a fixed public Sink target, not growing results."""

from __future__ import annotations

from _development_budget import check


class _Target:
    """One run owns a fresh sink wrapper and reuses only the allocated arrays."""

    def __init__(self, codes, scales, dtype):
        self.codes, self.scales = codes, scales
        self.dim, self.dtype = codes.shape[1], dtype
        self.next_row = 0

    def target(self, first_row, n_rows):
        if not self.next_row <= first_row < first_row + n_rows <= len(self.codes):
            raise ValueError("reused memory target reservation is outside the fixed output")
        codes = self.codes[first_row : first_row + n_rows]
        if self.scales is None:
            return codes
        return codes, self.scales[first_row : first_row + n_rows]

    def commit(self, first_row, rows):
        if first_row != self.next_row:
            raise ValueError("reused memory target commits are not ordered")
        codes = rows[0] if isinstance(rows, tuple) else rows
        self.next_row += len(codes)
        if self.next_row > len(self.codes):
            raise ValueError("reused memory target committed too many rows")

    def result(self):
        from fse import QuantizedRows

        if self.next_row != len(self.codes):
            raise ValueError("reused memory target has incomplete output")
        return self.codes if self.scales is None else QuantizedRows(self.codes, self.scales)


def operation(model, texts, job):
    import numpy as np
    from _development_encode import output_identity

    check(job.get("wall_deadline"))
    dtype = job["output_dtype"]
    codes = np.empty((len(texts), model.output_dim), dtype=dtype)
    scales = np.empty(len(texts), dtype=np.float32) if dtype == "int8" else None
    check(job.get("wall_deadline"))
    reference = model.encode(texts, workers=job["workers"], dtype=dtype)
    expected = output_identity(reference)
    del reference

    def encode():
        sink = _Target(codes, scales, dtype)
        model.encode(texts, workers=job["workers"], sink=sink)
        return sink.result()

    check(job.get("wall_deadline"))
    if output_identity(encode()) != expected:
        raise ValueError("reused memory Sink differs from regular encode output")
    return encode
