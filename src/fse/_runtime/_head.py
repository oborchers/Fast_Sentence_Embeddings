#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""A lone linear head in the runtime: a group of one (``fse._runtime._heads``).

``fse.Classifier`` and ``fse.Regressor`` run their text inference through the one runtime
of heads: ``make_head`` plans a group with one member and ``HeadPipeline`` is the thin
shell that keeps the interface of 2.0, with the sinks of a lone head and errors that name
no head (docs/architecture/runtime.md, "Runtime"; the rule ``linear@1``).
"""

from __future__ import annotations

from typing import Any

from numpy import ndarray

from fse import _codes
from fse._cpus import usable_cpus

from ._head_spec import HeadSpec
from ._heads import HeadsPipeline
from ._heads_plan import (
    FOLDED,
    NORMALIZED,
    STANDALONE,
    HeadsPlan,
    Member,
    plan_heads,
)
from ._pipeline import Pipeline, check_sink_rows, check_workers
from ._sinks import Sink, sink_dtype, wants_inverse_norms
from ._spec import EncodeSpec

# What a head gives per row: its logits, the probabilities of its link, or label indices.
OUTPUTS = ("logits", "proba", "labels")


def make_head(
    spec: EncodeSpec,
    head: HeadSpec,
    backend: str,
    *,
    own: Pipeline,
    with_vectors: bool = False,
    workers: int | None = None,
    isa: str | None = None,
) -> HeadPipeline:
    """The head on ``backend`` for the encoder of ``spec``, whose own pipeline there is
    ``own`` (its rows are the vectors): folded where that is legal and no vectors are
    wanted, else on the full mean, sharing ``own`` where the head reads the encoder's rows.
    The fold runs now, over ``workers`` threads (the usable CPUs by default); ``isa``
    selects the kernel of the cpu backend for the pipelines it makes."""
    plan = plan_heads(
        spec, (Member(None, head),), backend, own=own, with_vectors=with_vectors,
        workers=workers, isa=isa,
    )  # fmt: skip
    return HeadPipeline._of(plan)


class HeadPipeline:
    """A lone head on one backend. Made by ``make_head``, or from the pipeline of a folded
    table (``reader``: k columns, the bias still to add) by ``HeadPipeline(reader, head,
    folded=True)``."""

    def __init__(
        self,
        reader: Pipeline,
        head: HeadSpec,
        *,
        folded: bool,
        vectors: Pipeline | None = None,
        with_vectors: bool = False,
    ):
        if not folded or vectors is not None or with_vectors:
            raise ValueError("a head of the full rows is made by make_head")
        if reader.output_dim != head.k:
            raise ValueError(f"the reader gives rows of {reader.output_dim} values, not {head.k}")
        kind = FOLDED if head.weight is not None else STANDALONE
        plan = HeadsPlan(reader.spec, (Member(None, head),), (kind,), own=reader, table=reader)
        self._set(plan)

    @classmethod
    def _of(cls, plan: HeadsPlan) -> HeadPipeline:
        out = object.__new__(cls)
        out._set(plan)
        return out

    def _set(self, plan: HeadsPlan) -> None:
        self.plan = plan
        self.head = plan.members[0].head
        self.with_vectors = plan.with_vectors
        self._runtime = HeadsPipeline(plan)

    @property
    def runtime(self) -> HeadsPipeline:
        """The runtime of heads this head is a group of one of: the sinks and tees of a run
        are given to it as to any group."""
        return self._runtime

    @property
    def folded(self) -> bool:
        """Whether the head reads folded rows (k columns, the bias still to add)."""
        return self.plan.kinds[0] in (FOLDED, STANDALONE)

    @property
    def reader(self) -> Pipeline:
        """The pipeline whose pooling gives the rows the head reads."""
        plan = self.plan
        if self.folded:
            assert plan.table is not None
            return plan.table
        if plan.kinds[0] == NORMALIZED and not plan.with_vectors:
            assert plan.normed is not None
            return plan.normed
        return plan.full if plan.full is not None and plan.kinds[0] != NORMALIZED else plan.own

    @property
    def vectors(self) -> Pipeline | None:
        """The pipeline of the encoder's own rows where they are not the rows the head
        reads, else None (one pooling gives both)."""
        plan = self.plan
        if not plan.with_vectors:
            return None
        same = (plan.kinds[0] == NORMALIZED) == plan.encoder_normalizes
        return None if same else plan.own

    @property
    def vector_dim(self) -> int:
        return self.plan.own.output_dim

    def _row_bytes(self, output: str, vdtype: str, inverse: bool) -> int:
        """The bytes one row of a task takes at most (``HeadsPlan.row_bytes``)."""
        return self.plan.row_bytes((output,), vdtype, inverse)

    def infer(
        self,
        source: object,
        *,
        output: str,
        workers: int | None = None,
        sink: Sink | None = None,
        vector_sink: Sink | None = None,
        vector_dtype: str = "float32",
        chunk_bytes: int | None = None,
        first_row: int = 0,
    ) -> tuple[ndarray | None, Any]:
        """The ``output`` of every sentence of ``source`` (and, if the head was made with
        vectors, the vectors), returned or committed to ``sink`` (``vector_sink``), numbered
        from ``first_row`` there; errors name the rows from ``first_row``. Every argument
        is checked before the source is read."""
        workers = check_workers(workers) or usable_cpus()
        self._check_outputs(output, sink)
        vector_dtype, inverse = self._check_vectors(vector_sink, vector_dtype)
        results, vectors = self._runtime.infer(
            source,
            outputs=(output,),
            workers=workers,
            sinks=[sink],
            vector_sink=vector_sink,
            vector_dtype=vector_dtype,
            inverse=inverse,
            chunk_bytes=chunk_bytes,
            first_row=first_row,
        )
        return results[0], vectors

    def _check_outputs(self, output: str, sink: Sink | None) -> int:
        """The values per row of ``output``, checked with ``sink`` (float32, no labels)."""
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {', '.join(OUTPUTS)}, not {output!r}")
        width = self.head.width(output)
        if sink is not None:
            if output == "labels":
                raise ValueError("labels go to no sink; take them from the batches")
            check_sink_rows(sink, width)
            if sink_dtype(sink) != "float32" or wants_inverse_norms(sink):
                raise ValueError("a sink of head outputs takes float32 rows without inverse norms")
        return width

    def _check_vectors(self, vector_sink: Sink | None, dtype: str) -> tuple[str, bool]:
        """The storage dtype of the vectors and whether their target wants inverse norms."""
        if vector_sink is None:
            return _codes.check_dtype(dtype), False
        if not self.with_vectors:
            raise ValueError("a sink of vectors needs a head made with the vectors")
        check_sink_rows(vector_sink, self.vector_dim)
        return sink_dtype(vector_sink), wants_inverse_norms(vector_sink)
