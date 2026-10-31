#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The parameters of unigram@1, as a tokenizer.json gives them.

docs/architecture/tokenizers.md, "unigram@1". ``UnigramConfig`` holds the added tokens, the unknown id,
the normalizer steps, the Metaspace values and ``byte_fallback`` (the JSON parameters),
and the data beyond them: the float64 score of every piece and the charsmap of a
``precompiled`` step, as immutable bytes, so the config stays hashable and a pipeline
built from it cannot see a change. Values this fse does not support raise
UnsupportedArtifactError, malformed ones ValueError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy import ndarray

from fse._charsmap import Charsmap
from fse._contract import UnsupportedArtifactError

UNIGRAM = "unigram@1"
METASPACE = (("prepend_scheme", "always"), ("replacement", "\u2581"), ("split", False))
# The keys of every kind of step, and the classes of runs.
STEP_KEYS = {
    "precompiled": {"type"},
    "replace": {"type", "pattern", "content"},
    "runs": {"type", "class", "min", "content"},
    "strip": {"type", "left", "right"},
}
RUN_CLASSES = ("space", "white_space")
_INT_LIMIT = 2**62


def _exact_int(value: object, name: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, not {type(value).__name__}")
    if not low <= value < _INT_LIMIT:
        raise ValueError(f"{name} must be between {low} and 2**62 - 1, not {value}")
    return value


def _text(value: object, name: str, *, empty: bool = True) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{name} must be a{'' if empty else ' non-empty'} str, not {value!r}")
    value.encode("utf-8")  # a lone surrogate raises
    return value


def _flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, not {type(value).__name__}")
    return value


def _added(value: object) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(pair, tuple) and len(pair) == 2 for pair in value
    ):
        raise ValueError("added_tokens must be a tuple of (id, content) pairs")
    ids = [_exact_int(i, "the id of an added token", 0) for i, _ in value]
    contents = [_text(c, "the content of an added token", empty=False) for _, c in value]
    if ids != sorted(set(ids)):
        raise ValueError(f"added_tokens must be sorted by id, each id once, not {ids}")
    if len(set(contents)) != len(contents):
        raise ValueError("the contents of the added tokens must be unique")


def _step(step: object) -> None:
    """One normalizer step: a tuple of its (key, value) pairs, sorted by key."""
    if not isinstance(step, tuple) or not all(isinstance(p, tuple) and len(p) == 2 for p in step):
        raise ValueError(f"a normalizer step must be a tuple of (key, value) pairs, not {step!r}")
    items = dict(step)
    kind = items.get("type")
    if kind not in STEP_KEYS:
        raise UnsupportedArtifactError(
            f"unigram@1 has no normalizer step {kind!r}; it has {', '.join(STEP_KEYS)}"
        )
    if set(items) != STEP_KEYS[kind] or [k for k, _ in step] != sorted(items):
        raise ValueError(f"the step {kind} holds exactly {sorted(STEP_KEYS[kind])}, sorted")
    if kind == "replace":
        _text(items["pattern"], "the pattern of replace", empty=False)
        _text(items["content"], "the content of replace")
    elif kind == "runs":
        if items["class"] not in RUN_CLASSES:
            raise ValueError(f"the class of runs is one of {RUN_CLASSES}, not {items['class']!r}")
        _exact_int(items["min"], "the min of runs", 1)
        _text(items["content"], "the content of runs")
    elif kind == "strip":
        _flag(items["left"], "left of strip")
        _flag(items["right"], "right of strip")


def _scores(scores: object) -> int:
    if not isinstance(scores, bytes) or not scores or len(scores) % 8:
        raise ValueError("scores must be the bytes of one little-endian float64 per piece")
    if not np.isfinite(np.frombuffer(scores, dtype="<f8")).all():
        raise ValueError("every score must be finite")
    return len(scores) // 8


def step_of(value: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """A step of the JSON parameters as ``UnigramConfig`` holds it."""
    return tuple(sorted(value.items()))


@dataclass(frozen=True)
class UnigramConfig:
    """The parameters of ``unigram@1`` (docs/architecture/tokenizers.md, "unigram@1") and its data."""

    identifier: ClassVar[str] = UNIGRAM

    added_tokens: tuple[tuple[int, str], ...]
    unk_id: int
    normalizer: tuple[tuple[tuple[str, Any], ...], ...]
    scores: bytes
    charsmap: bytes | None = None
    byte_fallback: bool = False
    metaspace: tuple[tuple[str, Any], ...] = METASPACE

    def __post_init__(self):
        _added(self.added_tokens)
        _exact_int(self.unk_id, "unk_id", 0)
        if not isinstance(self.normalizer, tuple):
            raise ValueError("normalizer must be a tuple of steps")
        for step in self.normalizer:
            _step(step)
        precompiled = sum(dict(step)["type"] == "precompiled" for step in self.normalizer)
        if precompiled > 1:
            raise UnsupportedArtifactError("unigram@1 takes at most one precompiled step")
        if (self.charsmap is not None) != bool(precompiled):
            raise ValueError("a charsmap is given exactly with a precompiled step")
        if self.charsmap is not None:
            Charsmap(self.charsmap)
        n = _scores(self.scores)
        if self.unk_id >= n:
            raise ValueError(f"unk_id is {self.unk_id}, but there are {n} pieces")
        if _flag(self.byte_fallback, "byte_fallback"):
            raise UnsupportedArtifactError("unigram@1 with byte_fallback is not supported")
        if self.metaspace != METASPACE:
            raise UnsupportedArtifactError(
                f"unigram@1 supports the Metaspace {dict(METASPACE)}, not {dict(self.metaspace)}"
            )

    def steps(self) -> list[dict[str, Any]]:
        """The normalizer steps as JSON objects."""
        return [dict(step) for step in self.normalizer]

    def scores_array(self) -> ndarray:
        """The scores, a read-only float64 vector of one per piece."""
        return np.frombuffer(self.scores, dtype="<f8")

    def to_json(self) -> dict[str, Any]:
        """The parameters as a JSON object (the scores and the charsmap are tensors)."""
        return {
            "added_tokens": [list(pair) for pair in self.added_tokens],
            "byte_fallback": self.byte_fallback,
            "metaspace": dict(self.metaspace),
            "normalizer": self.steps(),
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_json(cls, value: object, scores: bytes, charsmap: bytes | None) -> UnigramConfig:
        """The parameters of a JSON object with exactly the keys of ``to_json``."""
        names = {"added_tokens", "byte_fallback", "metaspace", "normalizer", "unk_id"}
        if not isinstance(value, dict) or set(value) != names:
            raise ValueError(f"the parameters of unigram@1 are a JSON object of {sorted(names)}")
        added, steps, metaspace = value["added_tokens"], value["normalizer"], value["metaspace"]
        if not isinstance(added, list) or not all(isinstance(p, list) for p in added):
            raise ValueError("added_tokens must be a list of [id, content] pairs")
        if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
            raise ValueError("normalizer must be a list of JSON objects")
        if not isinstance(metaspace, dict):
            raise ValueError("metaspace must be a JSON object")
        return cls(
            tuple(tuple(pair) for pair in added),
            value["unk_id"],
            tuple(step_of(step) for step in steps),
            scores,
            charsmap,
            value["byte_fallback"],
            tuple(sorted(metaspace.items())),
        )

    def native_arguments(self) -> tuple:
        """The arguments of the native UnigramState after its table and vocabulary, and
        before its cuts: the scores, the unknown id, the ids of the added tokens, the steps
        (strings as UTF-8), the parts of the charsmap and the replacement of Metaspace."""
        steps: list[tuple[Any, ...]] = []
        for step in self.steps():
            kind = step["type"]
            if kind == "precompiled":
                steps.append((kind,))
            elif kind == "replace":
                steps.append((kind, step["pattern"].encode(), step["content"].encode()))
            elif kind == "runs":
                steps.append((kind, step["class"], step["min"], step["content"].encode()))
            else:
                steps.append((kind, step["left"], step["right"]))
        charsmap = None if self.charsmap is None else Charsmap(self.charsmap).parts()
        added = tuple(i for i, _ in self.added_tokens)
        replacement = dict(self.metaspace)["replacement"].encode()
        return self.scores_array(), self.unk_id, added, tuple(steps), charsmap, replacement

    def check_vocabulary(self, vocab_bytes: ndarray, vocab_offsets: ndarray) -> None:
        """One score per token of the vocabulary, and every added token is the
        vocabulary string of its id."""
        n = len(vocab_offsets) - 1
        if n != len(self.scores) // 8:
            raise ValueError(f"there are {len(self.scores) // 8} scores for {n} tokens")
        data = np.asarray(vocab_bytes, dtype=np.uint8)
        for i, content in self.added_tokens:
            if i >= n:
                raise ValueError(f"the added token {content!r} has id {i} of {n} tokens")
            word = data[int(vocab_offsets[i]) : int(vocab_offsets[i + 1])].tobytes()
            if word != content.encode("utf-8"):
                raise ValueError(f"the added token {content!r} has id {i}, whose token is {word!r}")
