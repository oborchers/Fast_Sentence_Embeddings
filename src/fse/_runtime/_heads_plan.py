#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The plan of a group of heads on one backend: which rows each head reads and from which
pipeline (docs/architecture/runtime.md, "Runtime", Groups of heads).

A group adds no arithmetic. Every head computes with the functions of its own path; the
group shares what does not depend on the head: the tokenization of a task, the pooling of
one folded table that holds the folded rows of all raw heads side by side (the pooling is
defined per value, so each column has the bits of a table of that head alone), and the
normalization of the pooled full rows. A head is read by one of four kinds of rows:

- ``folded``: columns of the shared folded table, the bias still to add;
- ``standalone``: the folded table of the encoder itself (all heads share it);
- ``normalized``: the pooled full rows after ``normalize@1``;
- ``raw``: the pooled full rows of a head on raw means that is run unfolded, because the
  vectors are asked for too.

A single head is a group of one: the same plan, the same bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fse._backends import get_backend
from fse._contract import Head, Link, Normalize
from fse._cpus import usable_cpus

from ._head_common import fold
from ._head_spec import HeadSpec
from ._pipeline import STEP_ROW_BYTES, Pipeline
from ._spec import EncodeSpec

FOLDED, STANDALONE, NORMALIZED, RAW = "folded", "standalone", "normalized", "raw"
# How the vectors come with the heads: none asked, the rows of the encoder's own pipeline
# (``own``), or the pooled raw rows of another pipeline normalized in a copy (``post``).
NO_VECTORS, OWN_ROWS, POST_ROWS = "none", "own", "post"


@dataclass(frozen=True, eq=False)
class Member:
    """A head of a group; ``name`` is None for a lone head (its errors name no head)."""

    name: str | None
    head: HeadSpec


def kind_of(head: HeadSpec, with_vectors: bool) -> str:
    """Which rows ``head`` reads, with or without the vectors."""
    if head.weight is None:
        return STANDALONE
    if head.normalized:
        return NORMALIZED
    return RAW if with_vectors else FOLDED


class HeadsPlan:
    """The members of a group, the pipelines that read their rows and how the vectors come.

    ``table`` reads the folded rows (K_F columns) or the standalone rows (k columns),
    ``normed`` the normalized full rows pooled on their own, ``full`` the raw full rows
    (``vector_mode`` tells how the vectors come when they are asked for), ``own`` is the
    pipeline of the encoder. ``primary`` plans the tasks and makes the batch of every task:
    all pipelines share the tokenizer and the vocabulary."""

    def __init__(
        self,
        spec: EncodeSpec,
        members: tuple[Member, ...],
        kinds: tuple[str, ...],
        *,
        own: Pipeline,
        table: Pipeline | None = None,
        full: Pipeline | None = None,
        normed: Pipeline | None = None,
        vector_mode: str = NO_VECTORS,
        workers: int = 1,
        isa: str | None = None,
    ):
        self.spec, self.members, self.kinds, self.own = spec, members, kinds, own
        self.backend_name, self.isa = own.backend.name, isa
        self.table, self.full, self.normed = table, full, normed
        self.vector_mode, self.workers = vector_mode, workers
        self.with_vectors = vector_mode != NO_VECTORS
        self.named = any(m.name is not None for m in members)
        self.encoder_normalizes = tuple(spec.steps) == (Normalize(),)
        self.starts = _starts(members, kinds)
        self._precompute()
        pipes = [p for p in (table, full, normed, own if self.with_vectors else None) if p]
        self.primary = max(pipes, key=lambda p: p.output_dim)

    def _precompute(self) -> None:
        """What a call and a task would otherwise recompute: the folded heads, the heads
        that read the shared table as it is, the names of the worker buffers and the bytes
        of a row that do not depend on the call."""
        self.folded = self.indices(FOLDED)
        self.has_normalized = bool(self.indices(NORMALIZED))
        # A lone folded head (or a standalone one) reads the table as it is, with no columns.
        self.direct = tuple(
            kind == STANDALONE or (kind == FOLDED and len(self.folded) == 1) for kind in self.kinds
        )
        count = len(self.members)
        self.logit_buffers = tuple(f"head_logits{i}" for i in range(count))
        self.column_buffers = tuple(f"head_columns{i}" for i in range(count))
        self.tags = tuple(f"{i}" for i in range(count))
        total = 4 * (self.table.output_dim if self.table else 0)
        if len(self.folded) > 1:
            total += 4 * sum(self.members[i].head.k for i in self.folded)
        full = 4 * self.spec.output_dim
        if self.indices(NORMALIZED, RAW):
            total += full + (STEP_ROW_BYTES * self.spec.output_dim if self.has_normalized else 0)
            total += full if self.has_normalized and self.indices(RAW) else 0
        self._fixed_bytes = total

    @property
    def folded_columns(self) -> int:
        """K_F, the columns of the shared folded table."""
        return sum(m.head.k for m, k in zip(self.members, self.kinds, strict=True) if k == FOLDED)

    def indices(self, *kinds: str) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.kinds) if kind in kinds)

    def who(self, *kinds: str, vectors: bool = False) -> str:
        """The heads of ``kinds`` (and the vectors) as an error names them."""
        names = [f"'{self.members[i].name}'" for i in self.indices(*kinds)]
        parts = [("head " if len(names) == 1 else "heads ") + ", ".join(names)] if names else []
        return " and ".join(parts + ["the vectors"] * (vectors and self.with_vectors))

    # --- the memory of a row --------------------------------------------------------

    def row_bytes(
        self,
        outputs: tuple[str, ...],
        vdtype: str,
        inverse: bool,
        extras: tuple[str | None, ...] | None = None,
        tee_bytes: int = 0,
        report_bytes: int = 0,
    ) -> int:
        """The bytes one row of a task takes at most: the rows the heads read (and their
        step), the logits and output targets of every head and, with the vectors, their
        target and scratch as the encoder's pipeline counts them. For one head the sum of
        the former single-head pipeline. A head may give one more output (``extras``, one
        entry per head: "logits", "proba" or None) into a target of its own, and a run that
        copies its outputs on (a score sink) holds ``tee_bytes`` more per row (the matrix of
        the sink and the document IDs); without them the sum is the one above. A length report
        of ``lengths="per_row"`` holds ``report_bytes`` per row until the commit of its task,
        counted once whatever the vectors."""
        total = self._fixed_bytes + tee_bytes + report_bytes
        for member, output in zip(self.members, outputs, strict=True):
            total += _head_bytes(member.head, output)
        for member, extra in zip(self.members, extras or (), strict=False):
            total += 0 if extra is None else 4 * member.head.width(extra)
        if self.with_vectors:
            total += self.own._row_bytes(vdtype, inverse)
        return total


def _head_bytes(head: HeadSpec, output: str) -> int:
    """The logits, the output and the probabilities of the label rule of one row."""
    multi = head.link is Link.MULTILABEL
    out = (head.k if multi else 8) if output == "labels" else 4 * head.width(output)
    total = 4 * head.k + out
    if output == "labels" and head.rule is Head.LINEAR2:
        total += 4 * head.width("proba")  # the probabilities the label is decided on
    return total


def _starts(members: tuple[Member, ...], kinds: tuple[str, ...]) -> tuple[int, ...]:
    """The first column of every folded head in the shared table (0 for the others)."""
    starts, at = [], 0
    for member, kind in zip(members, kinds, strict=True):
        starts.append(at if kind == FOLDED else 0)
        at += member.head.k if kind == FOLDED else 0
    return tuple(starts)


def plan_heads(
    spec: EncodeSpec,
    members: tuple[Member, ...],
    backend: str,
    *,
    own: Pipeline,
    with_vectors: bool = False,
    workers: int | None = None,
    isa: str | None = None,
) -> HeadsPlan:
    """The plan of ``members`` on ``backend`` for the encoder of ``spec`` whose own pipeline
    there is ``own`` (its rows are the vectors). The shared fold runs now, over ``workers``
    threads (the usable CPUs by default); ``isa`` selects the kernel of the cpu backend for
    the pipelines made."""
    if own.spec is not spec:
        raise ValueError("own must be the pipeline of spec")
    if not members:
        raise ValueError("a group needs at least one head")
    kinds = tuple(kind_of(m.head, with_vectors) for m in members)
    if with_vectors and STANDALONE in kinds:
        raise ValueError("a standalone head has no vectors: its table is folded")
    if STANDALONE in kinds and set(kinds) != {STANDALONE}:
        raise ValueError("standalone heads and complete heads cannot be run together")
    threads = workers or usable_cpus()
    maker = _Maker(spec, members, kinds, backend, own, isa, threads)
    full, mode = maker.full(with_vectors)
    return HeadsPlan(
        spec,
        members,
        kinds,
        own=own,
        table=own if STANDALONE in kinds else maker.table(),
        full=full,
        normed=maker.normed(with_vectors),
        vector_mode=mode,
        workers=threads,
        isa=isa,
    )


class _Maker:
    """The pipelines of a plan (the work of ``plan_heads``)."""

    def __init__(self, spec, members, kinds, backend, own, isa, workers):
        self.spec, self.members, self.kinds, self.backend = spec, members, kinds, backend
        self.own, self.isa, self.workers = own, isa, workers
        self.named = any(m.name is not None for m in members)
        self.normalizes = tuple(spec.steps) == (Normalize(),)

    def pipeline(self, **kwargs) -> Pipeline:
        return Pipeline(
            self.spec, self.backend, isa=self.isa, vocabulary=self.own.vocabulary, **kwargs
        )

    def table(self) -> Pipeline | None:
        """The pipeline of the shared folded table of the raw heads."""
        folded = [i for i, kind in enumerate(self.kinds) if kind == FOLDED]
        if not folded:
            return None
        weight = np.concatenate([self.members[i].head.weight for i in folded])
        host = get_backend(self.backend, isa=self.isa)
        try:
            pool = fold(host, self.spec.pool_input(), weight, self.workers)
        except ValueError as error:
            raise self.fold_error(host, folded, error) from None
        return self.pipeline(pool=pool, steps=())

    def fold_error(self, host, folded, error: ValueError) -> ValueError:
        """The fold of the group failed: with several heads, the first that fails alone."""
        if not self.named:
            return error
        for i in folded:
            try:
                fold(host, self.spec.pool_input(), self.members[i].head.weight, self.workers)
            except ValueError as alone:
                text = str(alone).replace("the head does", f"head {self.members[i].name!r} does", 1)
                return ValueError(text)
        return error

    def full(self, with_vectors: bool) -> tuple[Pipeline | None, str]:
        """The pipeline of the full rows, and how the vectors come when they are asked for:
        the encoder's own rows, or the raw rows of a pipeline without steps normalized in a
        copy (the encoder normalizes and a head reads the raw rows)."""
        if not with_vectors:
            return None, NO_VECTORS
        if self.normalizes and RAW in self.kinds:
            return self.pipeline(steps=()), POST_ROWS
        return self.own, OWN_ROWS

    def normed(self, with_vectors: bool) -> Pipeline | None:
        """The pipeline of the normalized rows pooled on their own (no vectors)."""
        if with_vectors or NORMALIZED not in self.kinds:
            return None
        return self.own if self.normalizes else self.pipeline(steps=(Normalize(),))
