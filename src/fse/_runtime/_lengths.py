#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What the length rule of an encoder cut, on request (docs/architecture/runtime.md,
"Length reports").

``LengthSettings.of`` checks the keywords ``lengths``, ``count_truncated`` and
``on_truncation`` the same way for every method that tokenizes text. The workers record
the cuts in their native workspaces (``fse._native.Workspace``); the calling thread adds
the record of each task in commit order to a ``LengthCollector``, which makes the
``LengthReport``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from fse._text import new_workspace

LEVELS = ("summary", "per_row")
# The bytes one row of a task holds until its commit: the cuts (1), the unknown tokens
# dropped (8) and the tokens (8); with counting the tokens and characters cut (16 more).
_TASK_ROW_BYTES = 17
_COUNT_ROW_BYTES = 16


@dataclass(frozen=True, slots=True)
class LengthSettings:
    """The checked keywords of one call: what its workers record and whether a cut raises."""

    lengths: str | None
    count_truncated: bool
    raise_cut: bool

    @classmethod
    def of(
        cls, lengths: Any, count_truncated: Any, on_truncation: Any, method: str
    ) -> LengthSettings | None:
        """The settings of the keywords of ``method``, or None for the defaults (nothing
        recorded, a cut sentence kept). Raises a ``TypeError`` for a wrong type and a
        ``ValueError`` for a wrong value."""
        if lengths is not None and not isinstance(lengths, str):
            raise TypeError(f"{method}: {_levels(lengths)}")
        if lengths is not None and lengths not in LEVELS:
            raise ValueError(f"{method}: {_levels(lengths)}")
        if not isinstance(count_truncated, bool):
            raise TypeError(
                f"{method}: count_truncated must be a bool, not {type(count_truncated).__name__}"
            )
        if not isinstance(on_truncation, str):
            raise TypeError(
                f"{method}: on_truncation must be 'keep' or 'raise', "
                f"not {type(on_truncation).__name__}"
            )
        if on_truncation not in ("keep", "raise"):
            raise ValueError(
                f"{method}: on_truncation must be 'keep' or 'raise', not {on_truncation!r}"
            )
        if count_truncated and lengths is None:
            raise ValueError(f"{method}: count_truncated needs lengths='summary' or 'per_row'")
        if lengths is None and on_truncation == "keep":
            return None
        return cls(lengths, count_truncated, on_truncation == "raise")

    @property
    def watch(self) -> int:
        """What the native workspace records: 1 the cuts, 2 also how much."""
        return 2 if self.count_truncated else 1

    @property
    def store_rows(self) -> bool:
        return self.lengths == "per_row"

    @property
    def row_bytes(self) -> int:
        """The bytes per row that the record of a task holds until its commit."""
        if not self.store_rows:
            return 0
        return _TASK_ROW_BYTES + (_COUNT_ROW_BYTES if self.count_truncated else 0)

    def collector(
        self, first_row: int, rule: tuple[int | None, int | None]
    ) -> LengthCollector | None:
        """The collector of a run from ``first_row`` under ``rule``, or None if the call asked
        only for ``on_truncation="raise"`` and wants no report."""
        return None if self.lengths is None else LengthCollector(self, first_row, rule)

    def workspace(self) -> Any:
        """A native workspace that records for these settings."""
        return new_workspace(self.watch, self.raise_cut, self.store_rows)


def _levels(value: object) -> str:
    shown = type(value).__name__ if not isinstance(value, str) else repr(value)
    return f"lengths must be None, 'summary' or 'per_row', not {shown}"


def _frozen(array: ndarray) -> ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class LengthReport:
    """What the length rule of the encoder cut in the rows of one call (or one batch).

    The rule is ``max_tokens`` ids per sentence of its first ``max_chars`` code points
    (``None``: no rule). A sentence is *character truncated* if it has more than
    ``max_chars`` code points, even if only white space follows. It is *token truncated* if
    its text up to ``max_chars`` has more than ``max_tokens`` ids (counted before unknown
    ids are dropped). ``tokens`` is the length of the row that is averaged: unknown ids
    dropped, a FastText word without a table row counted as 1. ``empty`` is ``tokens == 0``.
    A row of zeros is not always an empty one (see docs/architecture/runtime.md).

    The arrays of one value per row exist for ``lengths="per_row"`` and are read-only;
    ``truncated_tokens`` and ``truncated_chars`` (and their totals) exist for
    ``count_truncated=True``. ``first_row`` is the row of the run where the report starts.
    """

    first_row: int
    n_rows: int
    lengths: str
    count_truncated: bool
    max_tokens: int | None
    max_chars: int | None
    n_char_truncated: int
    n_token_truncated: int
    n_truncated: int
    n_empty: int
    tokens_total: int
    dropped_total: int
    truncated_tokens_total: int | None = None
    truncated_chars_total: int | None = None
    char_truncated: ndarray | None = None
    token_truncated: ndarray | None = None
    empty: ndarray | None = None
    tokens: ndarray | None = None
    dropped_unknown: ndarray | None = None
    truncated_tokens: ndarray | None = None
    truncated_chars: ndarray | None = None

    def truncated_rows(self) -> ndarray:
        """The rows (numbered in the run, from ``first_row``) a rule cut, ascending."""
        if self.char_truncated is None or self.token_truncated is None:
            raise ValueError("truncated_rows needs a report of lengths='per_row'")
        cut = self.char_truncated | self.token_truncated
        return np.flatnonzero(cut).astype(np.int64) + self.first_row


class LengthCollector:
    """The records of the tasks of one run in commit order, and their ``LengthReport``.
    Used by the calling thread only."""

    def __init__(
        self, settings: LengthSettings, first_row: int, rule: tuple[int | None, int | None]
    ):
        self.settings = settings
        self.first_row = first_row
        self.rule = rule
        self.n_rows = 0
        self.sums = np.zeros(8, dtype=np.int64)
        self.parts: list[tuple] = []

    def add(self, n_rows: int, taken: tuple) -> None:
        """The record ``(sums, rows)`` of the next task, which has ``n_rows`` rows."""
        sums, rows = taken
        self.n_rows += n_rows
        self.sums += np.array(sums, dtype=np.int64)
        if rows is not None:
            self.parts.append(rows)

    def report(self) -> LengthReport:
        """The report of the tasks added so far."""
        settings = self.settings
        sums = [int(v) for v in self.sums]
        fields: dict[str, Any] = {}
        if settings.store_rows:
            fields = self._rows()
        if settings.count_truncated:
            fields.update(truncated_tokens_total=sums[6], truncated_chars_total=sums[7])
        return LengthReport(
            first_row=self.first_row,
            n_rows=self.n_rows,
            lengths=settings.lengths or "summary",
            count_truncated=settings.count_truncated,
            max_tokens=self.rule[0],
            max_chars=self.rule[1],
            n_char_truncated=sums[0],
            n_token_truncated=sums[1],
            n_truncated=sums[2],
            n_empty=sums[3],
            tokens_total=sums[4],
            dropped_total=sums[5],
            **fields,
        )

    def _rows(self) -> dict[str, Any]:
        """The arrays of one value per row, from the parts of the tasks."""

        def joined(k: int, dtype: Any) -> ndarray:
            parts = [p[k] for p in self.parts]
            return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype)

        cuts = joined(0, np.uint8)
        tokens = joined(2, np.int64)
        fields = {
            "char_truncated": _frozen((cuts & 1) != 0),
            "token_truncated": _frozen((cuts & 2) != 0),
            "empty": _frozen(tokens == 0),
            "tokens": _frozen(tokens),
            "dropped_unknown": _frozen(joined(1, np.int64)),
        }
        if self.settings.count_truncated:
            fields["truncated_tokens"] = _frozen(joined(3, np.int64))
            fields["truncated_chars"] = _frozen(joined(4, np.int64))
        return fields


def rule_of(spec: Any) -> tuple[int | None, int | None]:
    """The ``(max_tokens, max_chars)`` of the tokenizer spec of an encoder."""
    return spec.max_tokens, spec.max_chars
