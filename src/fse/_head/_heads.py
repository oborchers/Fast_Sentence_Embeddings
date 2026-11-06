#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.Heads``: several heads of one encoder scored in one pass over the texts.

A group holds named ``Classifier`` and ``Regressor`` objects that read the rows of the same
encoder. It tokenizes each text once, pools the rows the heads read once and gives every
head the bits of the same head alone. It is a read-only mapping and exists only at run time:
it has no file format; the heads are saved one by one.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Literal, NoReturn

from numpy import ndarray

from fse._runtime import HeadsPipeline, Member, plan_heads
from fse._validation import PositiveInt, validate_arguments

from . import _heads_run
from ._classifier import Classifier
from ._model import LinearModel
from ._outputs import HeadsBatch
from ._regressor import Regressor

_NAME = re.compile(r"[a-z][a-z0-9_]*", re.ASCII)
Path = Literal["folded", "normalized", "unfolded", "standalone"]


class Heads(Mapping):
    """Heads of one encoder, scored together: ``Heads(quality=q, toxic=t)``.

    The names are the keywords (``[a-z][a-z0-9_]*``) and the order of the heads is the
    order of the keywords. Every head must be a ``Classifier`` or a ``Regressor`` of the same
    encoder, which the encoder fingerprints tell, and all complete or all standalone. A
    group is read-only and not picklable.

    ``predict`` gives, for every head, the array its own ``predict`` gives, or the output you
    name; ``predict_batches`` does so batch by batch, with the vectors if asked. Each head has
    exactly the bits of that head alone on the same path. If one head cannot score a text,
    the call fails, and the error names the head."""

    __slots__ = ("_heads", "_lock", "_pipelines")
    _heads: Mapping[str, LinearModel]
    _lock: threading.Lock
    _pipelines: dict[tuple[str, bool], HeadsPipeline]

    def __init__(self, /, *args: Any, **heads: Any):
        if args:
            raise TypeError(
                "give the heads as keywords, Heads(name=model); a dict or a list of pairs is "
                "not taken (Heads(**d) is)"
            )
        _check(heads)
        object.__setattr__(self, "_heads", MappingProxyType(dict(heads)))
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_pipelines", {})

    # --- the mapping ------------------------------------------------------------------

    def __getitem__(self, name: str) -> Classifier | Regressor:
        return self._heads[name]  # type: ignore[return-value]

    def __iter__(self) -> Iterator[str]:
        return iter(self._heads)

    def __len__(self) -> int:
        return len(self._heads)

    @property
    def encoder_fingerprint(self) -> str:
        """The fingerprint of the encoder of the heads: its stored content identity (a model
        loaded from a file carries the value of the file; ``Encoder.verify`` checks the
        content). Equal fingerprints mean the same stored content."""
        return next(iter(self._heads.values())).encoder_fingerprint

    @property
    def _first(self) -> LinearModel:
        return next(iter(self._heads.values()))

    @property
    def _tokenizer(self) -> str:
        return self._first._tokenizer

    @property
    def _standalone(self) -> bool:
        return self._first.standalone

    # --- the runtime ------------------------------------------------------------------

    def _pipeline(self, backend: str, *, with_vectors: bool) -> HeadsPipeline:
        key = (backend, with_vectors)
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            with self._lock:  # made once, published whole; never held while predicting
                pipeline = self._pipelines.get(key)
                if pipeline is None:
                    first = self._first
                    own = first._encoder._pipeline(backend)
                    members = tuple(Member(n, m._head) for n, m in self._heads.items())
                    plan = plan_heads(
                        first._encoder._spec, members, backend, own=own, with_vectors=with_vectors
                    )
                    pipeline = self._pipelines[key] = HeadsPipeline(plan)
        return pipeline

    # --- use --------------------------------------------------------------------------

    @validate_arguments
    def paths(self, *, return_vectors: bool = False) -> Mapping[str, Path]:
        """The path every head takes: ``folded`` (its weights fold into the table, the bias is
        added after pooling), ``normalized`` (the full mean, normalized), ``unfolded`` (the full
        mean, because the vectors are asked for too) or ``standalone``. A head on raw means
        takes ``folded`` without vectors and ``unfolded`` with them; the two agree only
        within the rounding of the fold (docs/architecture/numerical-contract.md)."""
        if return_vectors and self._standalone:
            raise ValueError("a standalone group has no vectors: its table is folded")
        return MappingProxyType({n: _path(m, return_vectors) for n, m in self._heads.items()})

    @validate_arguments
    def predict(
        self,
        texts: Any,
        *,
        output: Any = None,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
    ) -> Mapping[str, ndarray]:
        """The outputs of every head for every text, by name, in head order. By default each
        head gives what its own ``predict`` gives (labels, or the values of a regressor).
        ``output`` is an output name for all heads (``labels``, ``proba``, ``decision``,
        ``values``) or a mapping of head names to names; unnamed heads keep their default.
        All checks run before a text is read."""
        return _heads_run.infer(self, texts, output, workers, backend)

    @validate_arguments
    def predict_batches(
        self,
        batches: Any,
        *,
        output: Any = None,
        return_vectors: bool = False,
        workers: PositiveInt | None = None,
        backend: str = "cpu",
        vector_sink: Any = None,
        score_sink: Any = None,
        doc_ids: bool = False,
    ) -> Iterator[HeadsBatch]:
        """``predict`` for a stream, batch by batch: ``batches`` yields finite lists of texts
        or Arrow arrays. Every batch gives one ``HeadsBatch`` with the ``outputs`` of every
        head and, with ``return_vectors=True``, the ``vectors`` (the rows of the encoder; the
        heads on raw means then take the unfolded path, see ``paths``). Checked before any
        batch is read.

        With a ``score_sink`` (``fse.ScoreSink``) the scores of every head are also written
        to a score store, whatever ``output`` returns; the call attaches the run to the sink
        before the first batch is read, so a head whose path differs from that of the store
        is refused. Each batch then reports ``durable_rows``, the rows of the run that are
        durable; a consumer of a queue confirms its input up to there, and up to
        ``sink.durable_rows`` once the iteration ended (the sink is flushed then).

        A ``vector_sink`` takes the vectors of the encoder, the rows of ``return_vectors=True``
        (which it needs: the heads on raw means then take the unfolded path, see ``paths``).
        It is a ``fse.ShardSink`` or any ``fse.Sink``, alone or with a ``score_sink`` in a
        second folder: one pass writes both, with the same ``doc_ids``. ``durable_rows`` is
        then the minimum of the counts of both sinks. The two folders are not one store: no
        file ties them, each is a valid prefix on its own after a crash, and rows beyond the
        confirmed ones may be in one folder only; a replay writes them again (at least once).
        Give both sinks the same ``shard_rows``; document IDs are not unique keys. Two shard
        stores that hold unequal numbers of rows take a run only with ``doc_ids``. With
        ``doc_ids=True`` the batches are ``(texts, doc_ids)`` pairs as for
        ``Encoder.encode_batches(doc_ids=True)``, stored with the rows; the sink must have
        been made with ``doc_ids=True``. Stopping early does not flush: close the sink."""
        return _heads_run.batches(
            self,
            batches,
            output,
            return_vectors,
            workers,
            backend,
            vector_sink,
            score_sink,
            doc_ids,
        )

    # --- a value ----------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("Heads is read-only")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Heads is read-only")

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "Heads cannot be pickled; save each head with save() and make the group again"
        )

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}={_describe(m)}" for n, m in self._heads.items())
        return f"Heads({parts})"


def _check(heads: dict[str, Any]) -> None:
    """The heads of a group: at least one, named by the pattern, each a Classifier or a
    Regressor, all complete or all standalone, all of the same encoder."""
    if not heads:
        raise ValueError("Heads needs at least one head: Heads(name=model)")
    for name, model in heads.items():
        _check_one(name, model)
    _check_standalone(heads)
    _check_encoder(heads)


def _check_one(name: str, model: Any) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError(f"the name {name!r} of a head must match [a-z][a-z0-9_]*")
    if not isinstance(model, Classifier | Regressor):
        raise TypeError(
            f"head {name!r} must be a Classifier or a Regressor, not {type(model).__name__}"
        )


def _check_standalone(heads: dict[str, Any]) -> None:
    standalone = [n for n, model in heads.items() if model.standalone]
    if standalone and len(standalone) != len(heads):
        complete = [n for n in heads if n not in standalone]
        raise ValueError(
            f"a group is all standalone or all complete: {standalone} are standalone, "
            f"{complete} are complete"
        )


def _check_encoder(heads: dict[str, Any]) -> None:
    (first, *rest) = heads
    expected = heads[first].encoder_fingerprint
    for name in rest:
        found = heads[name].encoder_fingerprint
        if found != expected:
            raise ValueError(
                f"the heads {first!r} and {name!r} read different encoders "
                f"({expected[:12]} and {found[:12]}); a standalone head holds its own folded "
                "table as its encoder, and a quantized and a float32 encoder are different "
                "encoders"
            )


def _path(model: LinearModel, vectors: bool) -> Path:
    if model.standalone:
        return "standalone"
    if model.normalized:
        return "normalized"
    return "unfolded" if vectors else "folded"


def _describe(model: LinearModel) -> str:
    kind = "Regressor" if model.TASK == "regression" else "Classifier"
    return f"{kind}({model.n_outputs} outputs, {model.link}, {_path(model, False)})"
