"""An optional adapter joining document IDs to the existing positional sink protocol."""

from __future__ import annotations

from typing import Any

from fse._stream import StreamSource
from fse._text._ids import IdSource


def check_id_mode(sink: Any, ids: Any) -> None:
    enabled = getattr(sink, "has_doc_ids", False)
    if (ids is not None) != enabled:
        raise ValueError("doc_ids must be supplied exactly when the sink has doc_ids=True")


def check_batch_mode(sink: Any, doc_ids: bool) -> None:
    """The ``doc_ids`` argument of ``encode_batches`` must match the sink."""
    enabled = getattr(sink, "has_doc_ids", False)
    if doc_ids and not enabled:
        raise ValueError(
            "doc_ids=True needs a sink with doc_ids; EncodedBatch holds no document IDs"
        )
    if enabled and not doc_ids:
        raise ValueError("the sink stores doc_ids; pass doc_ids=True and (texts, doc_ids) batches")


def infer_ids(
    pipeline: Any, source: Any, sink: Any, workers: Any, first_row: int, ids: Any
) -> None:
    with IdSource(ids) as values:
        if not isinstance(source, StreamSource) and len(source) != len(values):
            raise ValueError(f"doc_ids has {len(values)} rows for {len(source)} texts")
        adapter = _IdSink(sink, values, first_row)
        pipeline.infer(source, workers=workers, sink=adapter, first_row=first_row)
        if adapter.end != first_row + len(values):
            raise ValueError("doc_ids has more rows than the text stream")


class _IdSink:
    extra_row_bytes = 8

    def __init__(self, sink: Any, values: IdSource, base: int):
        self.sink, self.values, self.base = sink, values, base
        self.end = base
        self.pending: dict[int, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sink, name)

    def target(self, first_row: int, n_rows: int) -> Any:
        ids = self.values.block(first_row - self.base, first_row - self.base + n_rows)
        target = self.sink.target(first_row, n_rows)
        self.pending[first_row] = ids
        self.end = first_row + n_rows
        return target

    def commit(self, first_row: int, rows: Any) -> None:
        self.sink.commit(first_row, rows, doc_ids=self.pending.pop(first_row))
