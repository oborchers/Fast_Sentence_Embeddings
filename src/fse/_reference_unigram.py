#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The reference of unigram@1 (docs/architecture/tokenizers.md, "unigram@1").

Plain Python on code points and bytes that follows the specification step by step and
defines the ids. It uses the grapheme data of ``fse._unicode_graphemes`` (Unicode 17.0.0)
and the charsmap of ``fse._charsmap``, never ``unicodedata`` or a regex engine, and
Python floats, which are float64. They follow the rounding mode of the thread, so every
encode checks that it is round to nearest. It is slow and meant for tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import pairwise
from typing import Any

from fse import _unicode_graphemes as ucd
from fse._charsmap import Charsmap
from fse._reference_wordpiece import split_added

# tokenizers' K_UNK_PENALTY: an unknown code point scores this below the lowest piece.
UNK_PENALTY = 10.0
# A grapheme of fewer bytes is searched in the charsmap whole (normalizers/precompiled.rs).
GRAPHEME_BYTES = 6
_CONTROLS = ("Control", "CR", "LF")


@lru_cache(maxsize=1 << 16)
def grapheme_break(cp: int) -> str:
    """The Grapheme_Cluster_Break of a code point; Other when it has none."""
    return next((value for value, cps in ucd.BREAKS.items() if cp in cps), "Other")


@lru_cache(maxsize=1 << 16)
def _incb(cp: int) -> str | None:
    return next((value for value, cps in ucd.INCB.items() if cp in cps), None)


def _gb9c(cps: Sequence[int], i: int) -> bool:
    """GB9c: a consonant, then linkers and extends with at least one linker, before i."""
    linkers, j = 0, i - 1
    while j >= 0 and _incb(cps[j]) in ("Extend", "Linker"):
        linkers += _incb(cps[j]) == "Linker"
        j -= 1
    return linkers > 0 and j >= 0 and _incb(cps[j]) == "Consonant"


def _gb11(cps: Sequence[int], i: int) -> bool:
    """GB11: an Extended_Pictographic, extends, a ZWJ (at i - 1), then one at i."""
    j = i - 2
    while j >= 0 and grapheme_break(cps[j]) == "Extend":
        j -= 1
    return j >= 0 and cps[j] in ucd.EXTENDED_PICTOGRAPHIC


def _joined(cps: Sequence[int], i: int, regional: int) -> bool:
    """Whether no boundary lies before cps[i] (rules GB3 to GB13); ``regional`` is the
    number of regional indicators in a row that end at i - 1."""
    a, b = grapheme_break(cps[i - 1]), grapheme_break(cps[i])
    if (a, b) == ("CR", "LF"):
        return True
    if a in _CONTROLS or b in _CONTROLS:
        return False
    hangul = {("L", "L"), ("L", "V"), ("L", "LV"), ("L", "LVT"), ("LV", "V"), ("LV", "T")}
    hangul |= {("V", "V"), ("V", "T"), ("LVT", "T"), ("T", "T")}
    if (a, b) in hangul or b in ("Extend", "ZWJ", "SpacingMark") or a == "Prepend":
        return True
    if _incb(cps[i]) == "Consonant" and _gb9c(cps, i):
        return True
    if a == "ZWJ" and cps[i] in ucd.EXTENDED_PICTOGRAPHIC and _gb11(cps, i):
        return True
    return a == b == "Regional_Indicator" and regional % 2 == 1


def boundaries(cps: Sequence[int]) -> list[int]:
    """The indices i (0 < i < len) with a grapheme boundary before cps[i] (UAX #29)."""
    out, regional = [], 0
    for i in range(1, len(cps)):
        regional = regional + 1 if grapheme_break(cps[i - 1]) == "Regional_Indicator" else 0
        if not _joined(cps, i, regional):
            out.append(i)
    return out


def graphemes(text: str) -> list[str]:
    """The extended grapheme clusters of ``text``."""
    cuts = [0, *boundaries([ord(ch) for ch in text]), len(text)]
    return [text[a:b] for a, b in pairwise(cuts) if a < b]


def precompiled(text: str, charsmap: Charsmap) -> str:
    """The step ``precompiled``: a grapheme of fewer than 6 bytes becomes the value of its
    shortest prefix key, otherwise each code point becomes its value, or stays."""
    out = []
    for grapheme in graphemes(text):
        data = grapheme.encode("utf-8")
        value = charsmap.find(data) if len(data) < GRAPHEME_BYTES else None
        if value is not None:
            out.append(value)
            continue
        for ch in grapheme:
            found = charsmap.find(ch.encode("utf-8"))
            out.append(ch if found is None else found)
    return "".join(out)


def _member(class_: str):
    if class_ == "space":
        return lambda ch: ch == " "
    if class_ == "white_space":
        return lambda ch: ord(ch) in ucd.WHITE_SPACE
    raise ValueError(f"no class {class_!r}")


def runs(text: str, class_: str, minimum: int, content: str) -> str:
    """The step ``runs``: every maximal run of at least ``minimum`` code points of the
    class becomes ``content``."""
    member, out, i = _member(class_), [], 0
    while i < len(text):
        j = i
        while j < len(text) and member(text[j]):
            j += 1
        if j - i >= minimum:
            out.append(content)
            i = j
        elif j > i:
            out.append(text[i:j])
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def strip(text: str, left: bool, right: bool) -> str:
    """The step ``strip``: leading and trailing White_Space."""
    start, end = 0, len(text)
    while left and start < end and ord(text[start]) in ucd.WHITE_SPACE:
        start += 1
    while right and end > start and ord(text[end - 1]) in ucd.WHITE_SPACE:
        end -= 1
    return text[start:end]


def normalize(text: str, steps: Sequence[Mapping[str, Any]], charsmap: Charsmap | None) -> str:
    """Step 3: the steps in order (the ``normalizer`` of the parameters)."""
    for step in steps:
        kind = step["type"]
        if kind == "precompiled":
            if charsmap is None:
                raise ValueError("the step precompiled needs a charsmap")
            text = precompiled(text, charsmap)
        elif kind == "replace":
            text = text.replace(step["pattern"], step["content"])
        elif kind == "runs":
            text = runs(text, step["class"], step["min"], step["content"])
        elif kind == "strip":
            text = strip(text, step["left"], step["right"])
        else:
            raise ValueError(f"no step {kind!r}")
    return text


def metaspace(text: str, replacement: str = "\u2581") -> str:
    """Step 4: U+0020 becomes the replacement, which is put in front unless it is there;
    an empty piece stays empty."""
    text = text.replace(" ", replacement)
    return text if not text or text.startswith(replacement) else replacement + text


_ONE, _HALF_ULP, _THREE_QUARTERS = 1.0, 2.0**-53, 3 * 2.0**-54


def check_rounding() -> None:
    """Raise unless float arithmetic of this thread rounds to nearest (every other mode
    fails one of the three sums, which are computed at run time, not folded)."""
    nearest = (
        _ONE + _HALF_ULP == _ONE
        and -_ONE - _HALF_ULP == -_ONE
        and _ONE + _THREE_QUARTERS == _ONE + 2 * _HALF_ULP
    )
    if not nearest:
        raise RuntimeError("the reference of unigram@1 needs the rounding mode round to nearest")


def _utf8_length(lead: int) -> int:
    return 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4


class Model:
    """The pieces of a Unigram model in id order with their float64 scores."""

    def __init__(self, pieces: Sequence[str], scores: Sequence[float], unk_id: int):
        if len(pieces) != len(scores) or not 0 <= unk_id < len(pieces):
            raise ValueError("a model needs one score per piece and unk_id among them")
        check_rounding()
        self.ids = {piece.encode("utf-8"): i for i, piece in enumerate(pieces)}
        if len(self.ids) != len(pieces):
            raise ValueError("the pieces of a model are unique")
        self.scores = [float(score) for score in scores]
        self.unk_id = unk_id
        self.unk_score = min(self.scores) - UNK_PENALTY
        # Every prefix of a piece that ends on a code point boundary: a candidate is
        # extended while it is one.
        self.prefixes = {
            piece.encode("utf-8")[: len(piece[:k].encode("utf-8"))]
            for piece in pieces
            for k in range(1, len(piece) + 1)
        }

    @staticmethod
    def add(score: float, held: float) -> float:
        return score + held

    @staticmethod
    def better(offer: float, held: float) -> bool:
        """tokenizers keeps the first of equal offers."""
        return offer > held

    def joins(self, piece_id: int) -> bool:
        """tokenizers joins every node whose id is unk_id, a piece of that id too."""
        return piece_id == self.unk_id

    def offers(self, data: bytes, start: int) -> list[tuple[int, int, float]]:
        """(end, id, score) of the pieces that are a prefix of ``data[start:]``, shortest
        first, then the unknown node if no piece is the code point at ``start``."""
        out: list[tuple[int, int, float]] = []
        end, size = start, _utf8_length(data[start])
        while end < len(data):
            end += _utf8_length(data[end])
            key = data[start:end]
            if key not in self.prefixes:
                break
            piece_id = self.ids.get(key)
            if piece_id is not None:
                out.append((end, piece_id, self.scores[piece_id]))
        if not out or out[0][0] != start + size:
            out.append((start + size, self.unk_id, self.unk_score))
        return out

    def viterbi(self, piece: str) -> list[int]:
        """Step 5: the ids of one pre-token."""
        data = piece.encode("utf-8")
        best: list[tuple[float, int, int] | None] = [None] * (len(data) + 1)
        best[0] = (0.0, 0, -1)
        start = 0
        while start < len(data):
            held = best[start]
            if held is None:
                raise AssertionError("every code point boundary is reached")
            for end, piece_id, score in self.offers(data, start):
                offer = self.add(score, held[0])
                old = best[end]
                if old is None or self.better(offer, old[0]):
                    best[end] = (offer, start, piece_id)
            start += _utf8_length(data[start])
        return self.join(data, best)

    def join(self, data: bytes, best: Sequence[tuple[float, int, int] | None]) -> list[int]:
        """Back from the end: consecutive joined nodes become one string, which is the id
        of the piece it equals, else unk_id."""
        out: list[int] = []
        run, end = b"", len(data)
        while end > 0:
            node = best[end]
            if node is None:
                raise AssertionError("every end on the path is reached")
            _, start, piece_id = node
            if self.joins(piece_id):
                run = data[start:end] + run
            else:
                if run:
                    out.append(self.ids.get(run, self.unk_id))
                    run = b""
                out.append(piece_id)
            end = start
        if run:
            out.append(self.ids.get(run, self.unk_id))
        out.reverse()
        return out


def encode_ids(
    text: str,
    model: Model,
    *,
    added: Mapping[str, int],
    normalizer: Sequence[Mapping[str, Any]],
    charsmap: Charsmap | None = None,
    replacement: str = "\u2581",
    max_tokens: int | None = None,
    max_chars: int | None = None,
    drop_unknown: bool = True,
) -> list[int]:
    """The ids of one sentence (steps 1 to 6). ``added`` maps the added tokens to their
    ids, ``normalizer`` holds the steps of the parameters."""
    text.encode("utf-8")  # a lone surrogate raises
    check_rounding()
    if max_chars is not None:
        text = text[:max_chars]
    ids: list[int] = []
    for piece, token in split_added(text, added):
        if token is not None:
            ids.append(token)
            continue
        pre = metaspace(normalize(piece, normalizer, charsmap), replacement)
        if pre:
            ids += model.viterbi(pre)
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return [i for i in ids if i != model.unk_id] if drop_unknown else ids
