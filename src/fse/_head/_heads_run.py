#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The inference of a group of heads: texts to the outputs of every head, in arrays or
batches (docs/architecture/runtime.md, "Runtime", Groups of heads).

Texts run through the one runtime of heads, in the workers: one tokenization per task for
all heads. A group has no store mode: its results are in memory, unless a ``score_sink`` takes the
scores of every head for good (``_score_target``). A single head runs the same way
(``_lone_run``). The ``output`` rule resolves what each
head gives before any text is read.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from numpy import ndarray

from fse._encoder._doc_ids import check_switch
from fse._encoder._sources import opened
from fse._runtime import LengthSettings, check_workers, rule_of, streaming
from fse._store import VectorStore

from ._outputs import HeadsBatch
from ._positions_run import Stream, stream_batches
from ._positions_run import start as start_stream
from ._run import RUNTIME, check_backend, finish
from ._sink_run import SinkRun, run_batches, start

METHOD = "Heads.predict_batches"

# What a head can give: labels, probabilities or decisions of a classifier, values of a regressor.
OUTPUT_NAMES = ("labels", "proba", "decision", "values")


@dataclass(frozen=True, slots=True)
class OutputRule:
    """What the output rule needs of a head: its ``task`` ("classification" or "regression")
    and its ``link`` (None: no probabilities). A group of heads and a score store both give
    one per head, so neither builds a model for it."""

    task: str
    link: Any


def rules_of(models: Mapping[str, Any]) -> dict[str, OutputRule]:
    """The ``OutputRule`` of every model of a group."""
    return {name: OutputRule(model.TASK, model._content.link) for name, model in models.items()}


def resolve_outputs(rules: Mapping[str, OutputRule], output: Any) -> tuple[str, ...]:
    """The output of every head, in order, from ``output``: None (each head gives its
    ``predict``), one name for all heads, or a mapping of names to outputs (unnamed heads
    keep their default). Raises before anything is read."""
    given = _given(rules, output)
    return tuple(_checked(name, rule, given.get(name)) for name, rule in rules.items())


def _given(models: Mapping[str, Any], output: Any) -> dict[str, str]:
    if output is None:
        return {}
    if isinstance(output, str):
        return dict.fromkeys(models, _name(output))
    if not isinstance(output, Mapping):
        raise TypeError(
            f"output must be None, an output name or a mapping of head names to output names, "
            f"not {type(output).__name__}"
        )
    for key in output:
        if not isinstance(key, str):
            raise TypeError(f"the keys of output are head names (str), not {type(key).__name__}")
        if key not in models:
            raise ValueError(f"output names the head {key!r}, but the heads are {list(models)}")
    return {key: _name(value) for key, value in output.items()}


def _name(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"an output is one of {', '.join(OUTPUT_NAMES)}, not {value!r}")
    if value not in OUTPUT_NAMES:
        raise ValueError(f"output must be one of {', '.join(OUTPUT_NAMES)}, not {value!r}")
    return value


def _checked(name: str, rule: OutputRule, output: str | None) -> str:
    """``output`` for the head ``name`` (its default for None), if the head can give it."""
    regressor = rule.task == "regression"
    if output is None:
        return "values" if regressor else "labels"
    if (output == "values") != regressor:
        what = (
            "a regressor gives values"
            if regressor
            else "a classifier gives labels, proba or decision"
        )
        raise ValueError(
            f"head {name!r} cannot give {output!r}: {what}; name the output of each head in a mapping"
        )
    if output == "proba" and rule.link is None:
        raise ValueError(f"head {name!r} has no probabilities (link none); use decision")
    return output


def check_group_call(workers: Any, backend: Any, items: Any) -> None:
    check_workers(workers)
    check_backend(backend)
    if isinstance(items, VectorStore):
        raise ValueError(
            "a group of heads reads texts; take the outputs of a store from the predict_batches of "
            "each head, or read a score store with fse.ScoreStore"
        )


def infer(
    heads: Any,
    texts: Any,
    output: Any,
    workers: Any,
    backend: Any,
    settings: LengthSettings | None = None,
) -> Any:
    """The outputs of every head for every text of ``texts``, in head order; with a level
    of ``settings`` the pair of them and the ``LengthReport``."""
    check_group_call(workers, backend, texts)
    outputs = resolve_outputs(rules_of(heads), output)
    pipeline = heads._pipeline(backend, with_vectors=False)
    collector = None if settings is None else settings.collector(0, _rule(heads))
    with opened(texts, heads._tokenizer) as source:
        rows, _ = pipeline.infer(
            source,
            outputs=tuple(RUNTIME[o] for o in outputs),
            workers=workers,
            lengths=settings,
            collector=collector,
        )
    found = _outputs(heads, rows, outputs)
    return found if collector is None else (found, collector.report())


def _rule(heads: Any) -> tuple[int | None, int | None]:
    return rule_of(next(iter(heads.values()))._encoder._spec.tokenizer)


def batches(
    heads: Any,
    items: Any,
    output: Any,
    return_vectors: bool,
    workers: Any,
    backend: Any,
    vector_sink: Any = None,
    score_sink: Any = None,
    doc_ids: Any = False,
    settings: LengthSettings | None = None,
    positions: bool = False,
) -> Iterator[HeadsBatch]:
    """The checks of ``predict_batches`` now, then its batches as a generator. With a
    ``score_sink`` the run attaches to it here, after every argument is checked and before
    the first batch is read: the sink learns the path of each head. With ``positions=True``
    the batches are ``(texts, position)`` and the sinks resume (``_positions_run``)."""
    check_group_call(workers, backend, items)
    check_switch(doc_ids, METHOD)
    outputs = resolve_outputs(rules_of(heads), output)
    if return_vectors and heads._standalone:
        raise ValueError("a standalone group has no vectors: its table is folded")
    kinds = tuple(RUNTIME[o] for o in outputs)
    starter = start_stream if positions else start
    run = starter(heads, kinds, return_vectors, vector_sink, score_sink, doc_ids, METHOD)
    pipeline = heads._pipeline(backend, with_vectors=return_vectors)
    return _batches(heads, pipeline, iter(items), outputs, workers, run, doc_ids, settings)


@streaming
def _batches(
    heads,
    pipeline,
    items: Iterable[Any],
    outputs,
    workers,
    run: SinkRun | Stream | None = None,
    doc_ids=False,
    settings: LengthSettings | None = None,
) -> Iterator[HeadsBatch]:
    kinds = tuple(RUNTIME[o] for o in outputs)

    def make(row, n, rows, vectors, last, report):
        return HeadsBatch(
            first_row=row,
            n_rows=n,
            outputs=_outputs(heads, rows, outputs),
            vectors=vectors,
            durable_rows=last,
            lengths=report,
        )

    if isinstance(run, Stream):
        yield from stream_batches(
            pipeline, items, run, workers, METHOD, make, settings, _rule(heads)
        )
        return
    yield from run_batches(
        heads._tokenizer,
        pipeline,
        items,
        kinds,
        workers,
        run,
        doc_ids,
        METHOD,
        make,
        settings,
        _rule(heads),
    )


def _outputs(heads: Any, rows: list, outputs: tuple[str, ...]) -> Mapping[str, ndarray]:
    """The outputs as ``predict`` of each head gives them (labels from their indices, one
    column flattened where the head gives (n,)), read-only in head order."""
    return MappingProxyType(
        {
            name: finish(model, result, output)
            for (name, model), result, output in zip(heads.items(), rows, outputs, strict=True)
        }
    )
