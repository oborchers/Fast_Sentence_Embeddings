#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The schema of the scores a group of heads stores (docs/architecture/scores.md, "Schema").

``ScoreSchema.from_heads(heads, stored)`` reads what each head is (its model fingerprint, task,
rule, link, labels) and what it stores (``stored=``: the decision (logits) or the
probabilities of a classifier, the values of a regressor), and lays the heads side by side in
the columns of ``scores``. The path of each head is not known yet: a head on raw means runs
folded without the vectors and unfolded with them, which only ``Heads.predict_batches``
learns; ``with_paths`` completes the schema when a run attaches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy import ndarray

from fse._container import checksum
from fse._head import Heads
from fse._head._derive import proba_width
from fse._store._layout import SCORE_STORE_BYTES, root_text
from fse._store._scores_info import STORED, ScoreHead, ScoreInfo

PENDING = "pending"  # the path of a head before a run attaches
_LONGEST_PATH = "normalized"  # the longest path name: the size of the root is checked with it


@dataclass(frozen=True, eq=False)
class ScoreSchema:
    """What a group of heads stores: the encoder, the heads with their columns and the
    thresholds of the multilabel heads (the constant tensors of every shard)."""

    encoder: str
    heads: tuple[ScoreHead, ...]
    thresholds: Mapping[str, ndarray]

    @classmethod
    def from_heads(cls, heads: Any, stored: Any = None) -> ScoreSchema:
        """The schema of ``heads`` storing ``stored`` (None: the decision of every classifier
        and the values of every regressor; one name for all heads; or a mapping of head names
        to names). Raises before anything is created."""
        if not isinstance(heads, Heads):
            raise TypeError(f"heads must be a fse.Heads, not {type(heads).__name__}")
        choices = _stored(heads, stored)
        found, limits, column = [], {}, 0
        for (name, model), what in zip(heads.items(), choices, strict=True):
            head = _head(name, model, what, column)
            found.append(head)
            column = head.columns[1]
            if head.multilabel:
                limits[head.threshold_name] = _frozen(model._head.thresholds)
        schema = cls(heads.encoder_fingerprint, tuple(found), limits)
        schema.check_size()
        return schema

    @property
    def dim(self) -> int:
        return sum(head.width for head in self.heads)

    @property
    def columns(self) -> dict[str, slice]:
        """The columns of each head in ``scores``, by head name."""
        return {head.name: slice(*head.columns) for head in self.heads}

    def with_paths(self, paths: Mapping[str, str]) -> ScoreSchema:
        """The schema once the run knows the path of every head."""
        if set(paths) != {head.name for head in self.heads}:
            raise ValueError(f"the paths name the heads {list(paths)}, not the heads of the sink")
        return replace(self, heads=tuple(replace(h, path=paths[h.name]) for h in self.heads))

    def info(self, doc_ids: bool) -> ScoreInfo:
        return ScoreInfo(self.encoder, doc_ids, self.heads)

    def check_size(self) -> None:
        """The root of the store must stay within its limit; checked before anything is
        created, with the longest name of a path."""
        longest = replace(self, heads=tuple(replace(h, path=_LONGEST_PATH) for h in self.heads))
        size = len(root_text(longest.info(True))) + 1
        if size > SCORE_STORE_BYTES:
            raise ValueError(
                f"the heads need a store.json of {size} bytes; a score store holds at most "
                f"{SCORE_STORE_BYTES} bytes of it (fewer heads or labels)"
            )

    def threshold_checksums(self) -> dict[str, str]:
        """The checksums the writer will give the threshold tensors of every shard."""
        return {name: checksum(array) for name, array in self.thresholds.items()}


def _frozen(array: ndarray | None) -> ndarray:
    if array is None:
        raise ValueError("a multilabel head needs its thresholds")
    return np.ascontiguousarray(array, dtype=np.float32)


def _stored(heads: Mapping[str, Any], stored: Any) -> list[str]:
    """What each head stores, in head order, from ``stored``."""
    if stored is None:
        given: dict[str, str | None] = {}
    elif isinstance(stored, str):
        given = dict.fromkeys(heads, stored)
    elif isinstance(stored, Mapping):
        given = dict(stored)
        for key in given:
            if not isinstance(key, str) or key not in heads:
                raise ValueError(f"stored names the head {key!r}, but the heads are {list(heads)}")
    else:
        raise TypeError(
            "stored must be None, a name or a mapping of head names to names, "
            f"not {type(stored).__name__}"
        )
    return [_choice(name, model, given.get(name)) for name, model in heads.items()]


def _choice(name: str, model: Any, what: Any) -> str:
    regressor = model.TASK == "regression"
    if what is None:
        return "values" if regressor else "decision"
    if not isinstance(what, str) or what not in STORED:
        raise ValueError(f"stored must be one of {', '.join(STORED)}, not {what!r} (head {name!r})")
    if (what == "values") != regressor:
        raise ValueError(
            f"head {name!r} cannot store {what!r}: "
            + (
                "a regressor stores values"
                if regressor
                else "a classifier stores decision or proba"
            )
            + "; name what each head stores in a mapping"
        )
    if what == "proba" and model._content.link is None:
        raise ValueError(f"head {name!r} has no probabilities (link none); store decision")
    return what


def _head(name: str, model: Any, what: str, column: int) -> ScoreHead:
    content, spec = model._content, model._head
    width = proba_width(spec.k, spec.link) if what == "proba" else spec.k
    regression = model.TASK == "regression"
    return ScoreHead(
        name=name,
        model=model.fingerprint,
        task=model.TASK,
        rule=spec.rule.id,
        link=content.link_id,
        stored=what,
        path=PENDING,
        labels=None if regression else tuple(content.labels),
        width=width,
        columns=(column, column + width),
        flat=(content.flat if spec.k == 1 else None) if regression else None,
    )
