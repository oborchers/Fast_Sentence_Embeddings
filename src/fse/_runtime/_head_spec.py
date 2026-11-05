"""Immutable linear-head specification used by the runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from fse._contract import Link, Normalize, Step

F32 = np.float32


@dataclass(frozen=True, eq=False)
class HeadSpec:
    """A linear head: ``weight`` (k, d) float32 and ``bias`` (k,), or for a standalone head
    (whose table is folded already) no weight; ``normalized``: it reads normalized rows;
    ``link``: the link of its probabilities, or None. Owns read-only copies."""

    weight: ndarray | None
    bias: ndarray
    normalized: bool
    link: Link | None
    thresholds: ndarray | None = None

    def __post_init__(self):
        bias = _frozen(self.bias, "bias", 1)
        object.__setattr__(self, "bias", bias)
        if self.weight is not None:
            weight = _frozen(self.weight, "weight", 2)
            if len(weight) != len(bias):
                raise ValueError(f"weight has {len(weight)} rows, but bias {len(bias)} values")
            object.__setattr__(self, "weight", weight)
        elif self.normalized:
            raise ValueError("a standalone head reads raw means; it cannot be normalized")
        if not isinstance(self.normalized, bool):
            raise TypeError("normalized must be a bool")
        if self.link is not None and not isinstance(self.link, Link):
            raise TypeError("link must be a Link or None")

        if self.link is Link.MULTILABEL:
            given = (
                np.full(len(bias), 0.5, dtype=F32) if self.thresholds is None else self.thresholds
            )
            limits = _frozen(given, "thresholds", 1)
            if limits.shape != bias.shape or ((limits < 0) | (limits > 1)).any():
                raise ValueError("thresholds must hold one value in [0, 1] per output")
            object.__setattr__(self, "thresholds", limits)
        elif self.thresholds is not None:
            raise ValueError("thresholds are given only for multilabel heads")

    @property
    def k(self) -> int:
        return len(self.bias)

    @property
    def steps(self) -> tuple[Step, ...]:
        """The step of the rows the head reads unfolded."""
        return (Normalize(),) if self.normalized else ()

    def width(self, output: str) -> int:
        """The values per row of ``output``: k logits, 2 probabilities for k = 1 (else k),
        one label index."""
        if output == "labels":
            return self.k if self.link is Link.MULTILABEL else 1
        if output == "proba":
            if self.link is None:
                raise ValueError("the head has no link, so no probabilities")
            return 2 if self.k == 1 and self.link is not Link.MULTILABEL else self.k
        return self.k


def _frozen(value: object, name: str, ndim: int) -> ndarray:
    if not isinstance(value, ndarray) or value.dtype != F32 or value.ndim != ndim or not value.size:
        raise ValueError(f"{name} must be a non-empty float32 array of {ndim} dimensions")
    copy = np.array(value, copy=True, order="C")
    if not np.isfinite(copy).all():
        raise ValueError(f"{name} must be finite")
    copy.setflags(write=False)
    return copy
