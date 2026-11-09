#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""A resumed run of heads: ``predict(texts, vector_sink=, score_sink=, shard=)`` of
``fse.Heads``, ``Classifier`` and ``Regressor`` with sinks made with ``resume=True``
(docs/architecture/scores.md, "Resuming").

The call reads a source that can be read again, hashes it once and lets every sink compare
that identity with its store; a store from another source, partition or heads is refused
before any row is reserved. Then it runs the rows no sink holds. The two stores of a call
(vectors and scores) may hold different numbers of rows, because a run stopped between
their shards: the rows only the longer one holds are skipped, and the rows only the shorter
one lacks run in a first phase into that sink alone. A second phase runs the rest into both.
The longer sink counts its rows from 0 in the second phase, the shorter one goes on:
``_Shifted`` moves the rows of the run for it. Whatever phase a row runs in, it is computed
as the uninterrupted run computes it, bit for bit.

The outputs of the heads are not returned: they go to sinks that keep nothing, so the memory
of a call does not grow with the rows of the source.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from fse._encoder import _resume
from fse._encoder._batches import check_sink, flush, run_into
from fse._encoder._sources import opened
from fse._partition import partition
from fse._runtime import LengthSettings, check_workers, rule_of
from fse._text._ids import IdSource

from ._heads_run import resolve_outputs, rules_of
from ._run import check_backend
from ._score_target import STORED_RUNTIME, check_score_sink
from ._sink_run import Shifted, SinkRun


class _Discard:
    """A sink that keeps nothing: it hands out a buffer for each task and drops it at the
    commit. A run that keeps only the stored scores and vectors gives it the main outputs of
    the heads (and the vectors, if they are asked for but no store takes them)."""

    inverse_norms = False
    dtype = "float32"

    def __init__(self, width: int):
        self.dim = width

    def target(self, first_row: int, n_rows: int) -> ndarray:
        del first_row
        return np.empty((n_rows, self.dim), np.float32)

    def commit(self, first_row: int, rows: ndarray, *, doc_ids: Any = None) -> None:
        pass


@dataclass(frozen=True)
class Subject:
    """What a resumed run runs: the heads by name (a group, or one head under the name of
    its score sink), the way to make their pipeline, and the head the encoder is read from."""

    models: Mapping[str, Any]
    pipeline: Callable[[str, bool], Any]

    @property
    def first(self) -> Any:
        return next(iter(self.models.values()))


@dataclass
class _Member:
    role: str  # "vector" or "score"
    sink: Any
    kept: int = 0


@dataclass(frozen=True)
class _Phase:
    """The source rows ``[first, last)`` run into ``members`` (each with the shift of its
    rows), numbering the rows of the run from ``start``."""

    first: int
    last: int
    start: int
    members: tuple[tuple[_Member, int], ...]


# --- the arguments, checked before a text is read -----------------------------------------


def wanted(*given: Any) -> bool:
    """Whether resume sinks or source coordinates were given."""
    return any(value is not None for value in given)


def _members(vector_sink: Any, score_sink: Any) -> list[_Member]:
    given = [("vector", vector_sink), ("score", score_sink)]
    members = [_Member(role, sink) for role, sink in given if sink is not None]
    if not members:
        raise ValueError(
            "shard= and doc_ids= of predict need a vector_sink or a score_sink "
            "made with resume=True, such as fse.ScoreSink(path, heads, resume=True)"
        )
    for member in members:
        if not _resume.is_resume_sink(member.sink):
            raise ValueError(
                f"the {member.role}_sink must be made with resume=True: predict resumes its "
                "sinks (use predict_batches for a sink without resume)"
            )
    return members


def _check_ids(members: list[_Member], doc_ids: Any) -> None:
    if isinstance(doc_ids, bool):
        raise TypeError(
            "predict takes the document IDs themselves, one per text (a list of int, a "
            "one-dimensional int64 array or an Arrow int64 column), not a bool"
        )
    for member in members:
        if (doc_ids is not None) != (member.sink.has_doc_ids is True):
            raise ValueError(
                f"doc_ids must be supplied exactly when the sinks have doc_ids=True "
                f"(the {member.role}_sink has doc_ids={member.sink.has_doc_ids})"
            )


def _check_vectors(subject: Subject, members: list[_Member], return_vectors: bool) -> None:
    first = subject.first
    if return_vectors and first.standalone:
        raise ValueError("a standalone model has no vectors: its table is folded")
    for member in members:
        if member.role != "vector":
            continue
        if not return_vectors:
            raise ValueError(
                "vector_sink takes the vectors of return_vectors=True; pass return_vectors=True "
                "(a head on raw means then takes the unfolded path, see path() and Heads.paths)"
            )
        dim = first.encoder.output_dim
        hint = f"; the vector_sink takes the vectors of the encoder ({dim} values)"
        check_sink(member.sink, None, first.encoder_fingerprint, dim, hint, resumed=True)


def check_arguments(
    subject: Subject,
    texts: Any,
    output: Any,
    return_vectors: bool,
    workers: Any,
    backend: Any,
    vector_sink: Any,
    score_sink: Any,
    doc_ids: Any,
    shard: Any,
) -> tuple[list[_Member], tuple[int, int]]:
    """The sinks of the call as members and its partition, after the checks that read
    nothing of the source; raises on the first that fails."""
    part = _resume.check_shard(shard)
    members = _members(vector_sink, score_sink)
    for member in members:
        member.sink._check_open()
        if member.sink._ready:
            raise RuntimeError(
                f"the {member.role}_sink took a source already; make a new sink for another run"
            )
    check_workers(workers)
    check_backend(backend)
    resolve_outputs(rules_of(subject.models), output)
    if score_sink is not None:
        check_score_sink(score_sink, subject.models, resumed=True)
    _check_vectors(subject, members, return_vectors)
    _check_apart(vector_sink, score_sink)
    _check_ids(members, doc_ids)
    _resume.check_texts(texts)
    return members, part


def _check_apart(vector_sink: Any, score_sink: Any) -> None:
    """Two stores need two folders that do not contain each other."""
    if vector_sink is None or score_sink is None:
        return
    one, other = vector_sink.path.resolve(), score_sink.path.resolve()
    if one == other or one.is_relative_to(other) or other.is_relative_to(one):
        raise ValueError("vector_sink and score_sink must write to two folders apart")


# --- the phases ---------------------------------------------------------------------------


def phases(members: list[_Member], first: int, last: int) -> list[_Phase]:
    """The phases of a run of the part ``[first, last)`` of the source, given the rows each
    sink holds. One sink, or two that hold as many, run in one phase from the first row
    none of them holds. When one holds fewer, a first phase runs the rows between the two
    counts into it alone, and a second phase runs the rest into both."""
    low = min(m.kept for m in members)
    high = max(m.kept for m in members)
    found = []
    ahead = high - low  # the rows of the run before the second phase
    if ahead:
        found.append(
            _Phase(first + low, first + high, 0, tuple((m, 0) for m in members if m.kept == low))
        )
    shifted = tuple((m, 0 if m.kept == low else -ahead) for m in members)
    found.append(_Phase(first + high, last, ahead, shifted))
    return found


def _is_new(sink: Any) -> bool:
    return bool(sink._new) if hasattr(sink, "_new") else not sink._exists


def _attach(
    subject: Subject,
    members: list[_Member],
    identity: Any,
    part: tuple[int, int],
    seconds: float,
    return_vectors: bool,
) -> None:
    """Let every sink compare the identity with its store. Stores that exist go first: a
    mismatch is refused before a new store is made."""
    paths = {n: m.path(return_vectors=return_vectors) for n, m in subject.models.items()}
    first, last = partition(identity.rows, *part)
    for member in sorted(members, key=lambda m: _is_new(m.sink)):
        if member.role == "score":
            member.kept = member.sink._attach_source(identity, part, seconds, paths)
        else:
            member.kept = member.sink._attach_source(identity, part, seconds)
        if member.kept > last - first:
            raise ValueError(
                f"the store holds {member.kept} rows, more than the {last - first} rows of its part"
            )


@contextmanager
def _ids(values: Any) -> Iterator[IdSource | None]:
    if values is None:
        yield None
        return
    with IdSource(values) as opened_ids:
        yield opened_ids


@dataclass
class _Run:
    subject: Subject
    base: Any  # the source
    identity: Any
    ids: IdSource | None
    return_vectors: bool
    workers: Any
    backend: str
    settings: LengthSettings | None
    collector: Any


def _run_phase(run: _Run, phase: _Phase) -> None:
    """Run one phase: the window of its rows into its sinks."""
    if phase.first >= phase.last:
        return
    source = _resume.window(run.base, run.identity, phase.first, phase.last)
    ids = None if run.ids is None else run.ids.window(phase.first, phase.last)
    try:
        sinks = {m.role: Shifted(m.sink, delta) if delta else m.sink for m, delta in phase.members}
        if "score" in sinks:
            _heads(run, source, sinks, phase.start, ids)
        else:
            pipeline = run.subject.first._encoder._pipeline(run.backend)
            run_into(
                pipeline,
                source,
                sinks["vector"],
                run.workers,
                phase.start,
                ids,
                run.settings,
                run.collector,
            )
    finally:
        source.close()


def _heads(run: _Run, source: Any, sinks: dict[str, Any], start: int, ids: Any) -> None:
    """The heads over ``source`` into the score sink (and the vector sink, or a discarding
    one if the vectors are asked for and no store takes them in this phase)."""
    subject = run.subject
    vectors = run.return_vectors
    pipeline = subject.pipeline(run.backend, vectors)
    score = sinks["score"]
    kinds = tuple(STORED_RUNTIME[head.stored] for head in score._score_schema.heads)
    discard = [
        _Discard(member.head.width(kind))
        for member, kind in zip(pipeline.plan.members, kinds, strict=True)
    ]
    vector = sinks.get("vector") or (_Discard(pipeline.vector_dim) if vectors else None)
    batch = SinkRun(kinds, vector, score, ids is not None, discard)
    batch.infer(pipeline, source, kinds, run.workers, start, ids, run.settings, run.collector)


def resumed(
    subject: Subject,
    texts: Any,
    members: list[_Member],
    part: tuple[int, int],
    *,
    return_vectors: bool,
    workers: Any,
    backend: str,
    doc_ids: Any,
    settings: LengthSettings | None,
) -> Any:
    """The resumed run of ``texts`` into the sinks of ``members``: ``None``, or with
    ``settings`` of a level the pair ``(None, LengthReport)``."""
    first = subject.first
    with _ids(doc_ids) as ids, opened(texts, first._tokenizer) as base:
        began = time.perf_counter()
        identity = _resume.identify(base, ids)
        _attach(subject, members, identity, part, time.perf_counter() - began, return_vectors)
        start, end = partition(identity.rows, *part)
        row = start + min(m.kept for m in members)
        collector = (
            None
            if settings is None
            else settings.collector(row, rule_of(first._encoder._spec.tokenizer))
        )
        run = _Run(
            subject, base, identity, ids, return_vectors, workers, backend, settings, collector
        )
        try:
            for phase in phases(members, start, end):
                _run_phase(run, phase)
        except Exception as error:
            _resume.add_row_note(error, row)
            raise
        base._check_open()  # a file that changed during the run
    for member in sorted(members, key=lambda m: m.role != "vector"):  # vectors first
        flush(member.sink)
    return None if collector is None else (None, collector.report())


# --- the entries of Heads.predict and of the predict of one head ---------------------------


def predict_heads(
    heads: Any,
    texts: Any,
    output: Any,
    return_vectors: bool,
    workers: Any,
    backend: Any,
    vector_sink: Any,
    score_sink: Any,
    doc_ids: Any,
    shard: Any,
    settings: LengthSettings | None,
) -> Any:
    """``Heads.predict`` into resumable sinks."""
    subject = Subject(
        heads._heads, lambda kind, vectors: heads._pipeline(kind, with_vectors=vectors)
    )
    return _predict(
        subject,
        texts,
        output,
        return_vectors,
        workers,
        backend,
        vector_sink,
        score_sink,
        doc_ids,
        shard,
        settings,
    )


def predict_lone(
    model: Any,
    texts: Any,
    return_vectors: bool,
    workers: Any,
    backend: Any,
    vector_sink: Any,
    score_sink: Any,
    doc_ids: Any,
    shard: Any,
    settings: LengthSettings | None,
) -> Any:
    """``predict`` of a single head into resumable sinks; the score sink names the head."""
    name = "head"
    schema = getattr(score_sink, "_score_schema", None)
    if schema is not None:
        names = [head.name for head in schema.heads]
        if len(names) != 1:
            raise ValueError(
                f"the sink was made for the heads {names}; a single head writes a sink of one head"
            )
        name = names[0]
    subject = Subject(
        {name: model}, lambda kind, vectors: model._pipeline(kind, with_vectors=vectors).runtime
    )
    return _predict(
        subject,
        texts,
        None,
        return_vectors,
        workers,
        backend,
        vector_sink,
        score_sink,
        doc_ids,
        shard,
        settings,
    )


def _predict(
    subject: Subject,
    texts: Any,
    output: Any,
    return_vectors: bool,
    workers: Any,
    backend: Any,
    vector_sink: Any,
    score_sink: Any,
    doc_ids: Any,
    shard: Any,
    settings: LengthSettings | None,
) -> Any:
    members, part = check_arguments(
        subject,
        texts,
        output,
        return_vectors,
        workers,
        backend,
        vector_sink,
        score_sink,
        doc_ids,
        shard,
    )
    return resumed(
        subject,
        texts,
        members,
        part,
        return_vectors=return_vectors,
        workers=workers,
        backend=backend,
        doc_ids=doc_ids,
        settings=settings,
    )
