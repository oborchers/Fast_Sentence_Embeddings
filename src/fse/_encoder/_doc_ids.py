"""An optional adapter joining document IDs to the existing positional sink protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fse._stream import StreamSource
from fse._text._ids import IdSource, IdWindow


def refuse_score_sink(sink: Any) -> None:
    """A ScoreSink takes the scores of heads only, whatever its document-ID mode: it goes to
    ``score_sink=`` of ``predict_batches``, never as a sink of rows or vectors."""
    if hasattr(sink, "score_schema"):
        raise ValueError(
            "a ScoreSink stores the scores of heads, not the rows of an encoder, the vectors "
            "of a head or its outputs; pass it as score_sink= to predict_batches of a "
            "Classifier, a Regressor or fse.Heads"
        )


class ShardSinkMarker:
    """Base of the shard sinks of fse (``ShardSinkBase``). 2.0 had none that stores document
    IDs, so only these are refused when a call brings no IDs; a foreign sink never is."""


def id_mode(sink: Any) -> bool:
    """Whether the sink stores document IDs: only ``has_doc_ids`` exactly True says so.

    Any other value, also a truthy one, is what a sink of 2.0 may have set for itself; fse
    ignores it unless the call brings document IDs.
    """
    return getattr(sink, "has_doc_ids", None) is True


def own_id_sink(sink: Any) -> bool:
    """Whether the sink is one of fse's shard sinks in document-ID mode. 2.0 had no such
    sink, so it never meets a call without IDs. A foreign sink with ``has_doc_ids`` True
    ran in 2.0 as a plain sink and still does."""
    return id_mode(sink) and isinstance(sink, ShardSinkMarker)


def _refuse_other_value(sink: Any, role: str = "sink", method: str = "encode") -> None:
    """A call with IDs needs ``has_doc_ids`` True; a value that is no bool is a type error."""
    value = getattr(sink, "has_doc_ids", False)
    if not isinstance(value, bool):
        raise TypeError(
            f"{method}: has_doc_ids of the {role} must be True for doc_ids, not {type(value).__name__}"
        )


def check_switch(doc_ids: Any, method: str) -> bool:
    """The ``doc_ids`` switch of ``method``: a bool, else a ``TypeError``."""
    if not isinstance(doc_ids, bool):
        raise TypeError(f"{method}: doc_ids must be a bool, not {type(doc_ids).__name__}")
    return doc_ids


def check_modes(
    sinks: Sequence[tuple[str, Any]], doc_ids: bool, method: str, needs: str = "a sink"
) -> None:
    """``doc_ids`` of ``method`` against the sinks given, ``(role, sink)`` each.

    ``doc_ids=True`` needs a sink in document-ID mode for every sink given (``needs`` names
    what to give if there is none). Without it, a sink of fse in document-ID mode is refused,
    as it would fail at its first commit; a foreign sink with ``has_doc_ids`` True runs as
    in 2.0. Raises before anything is read."""
    if doc_ids and not sinks:
        raise ValueError(f"doc_ids=True needs {needs} with doc_ids; {method} has none")
    for role, sink in sinks:
        stores = id_mode(sink)
        if doc_ids and not stores:
            _refuse_other_value(sink, role, method)
            raise ValueError(
                f"doc_ids=True needs the {role} to have doc_ids; the {role} has none, "
                f"and {method} gives document IDs only to a sink made with doc_ids=True"
            )
        if not doc_ids and own_id_sink(sink):
            raise ValueError(
                f"the {role} stores doc_ids; pass doc_ids=True and (texts, doc_ids) batches "
                f"to {method}"
            )


def split_pair(k: int, item: Any, method: str) -> tuple[Any, Any]:
    """The ``(texts, doc_ids)`` of batch ``k`` of a call with ``doc_ids=True``."""
    if not isinstance(item, tuple) or len(item) != 2:
        raise TypeError(f"{method}: batch {k} must be a (texts, doc_ids) tuple")
    if item[1] is None:
        raise TypeError(f"{method}: batch {k} must give its doc_ids, not None")
    return item


def check_id_mode(sink: Any, ids: Any) -> None:
    refuse_score_sink(sink)
    if isinstance(ids, bool):
        raise TypeError(
            "encode takes the document IDs themselves, one per text (a list of int, a "
            "one-dimensional int64 array or an Arrow int64 column), not a bool; "
            "doc_ids=True is the switch of encode_batches"
        )
    enabled = id_mode(sink)
    if ids is not None and not enabled:
        _refuse_other_value(sink)
    if ids is None and not own_id_sink(sink):
        return  # a plain sink, or a foreign one that ran so in 2.0
    if (ids is not None) != enabled:
        raise ValueError("doc_ids must be supplied exactly when the sink has doc_ids=True")


def infer_ids(
    pipeline: Any,
    source: Any,
    sink: Any,
    workers: Any,
    first_row: int,
    ids: Any,
    lengths: Any = None,
    collector: Any = None,
) -> None:
    if isinstance(ids, IdWindow):  # a resumed run: its IDs are open already
        _run_ids(pipeline, source, sink, workers, first_row, ids, lengths, collector)
        return
    with IdSource(ids) as values:
        _run_ids(pipeline, source, sink, workers, first_row, values, lengths, collector)


def _run_ids(
    pipeline: Any,
    source: Any,
    sink: Any,
    workers: Any,
    first_row: int,
    values: Any,
    lengths: Any,
    collector: Any,
) -> None:
    if not isinstance(source, StreamSource) and len(source) != len(values):
        raise ValueError(f"doc_ids has {len(values)} rows for {len(source)} texts")
    adapter = _IdSink(sink, values, first_row)
    pipeline.infer(
        source,
        workers=workers,
        sink=adapter,
        first_row=first_row,
        lengths=lengths,
        collector=collector,
    )
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
