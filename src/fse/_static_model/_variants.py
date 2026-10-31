#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The encoders of one model under different rules (ARCHITECTURE.md, "model2vec
compatibility").

model2vec lets every ``encode`` call state another ``max_length`` or ``normalize``. An
fse ``Encoder`` is immutable, so each pair of the two is an encoder of its own: the same
table and the same vocabulary table, a tokenizer with another length rule or another
step. They are made on first use and kept in a bounded cache (a per-call ``max_length``
is unbounded), the least recently used going first; the model's own pair never leaves it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from fse._encoder import Encoder
from fse._encoder._format import Source
from fse._encoder._model2vec_rules import Rules
from fse._encoder._persist import converted
from fse._text import Vocabulary, make_vocabulary
from fse._tokenizer_spec import TokenizerSpec

LIMIT = 8


class Variants:
    """The encoders of one model, by ``(max_length, normalize)`` (``max_length`` None: no
    cut). Several threads may ask at once."""

    def __init__(self, rules: Rules, limit: int = LIMIT):
        self._rules = rules
        self._limit = limit
        self._lock = threading.Lock()
        self._vocabulary: Vocabulary | None = None
        self._made: OrderedDict[tuple[int | None, bool], Encoder] = OrderedDict()
        self._own: tuple[int | None, bool] | None = None

    def own(self, max_length: int | None, normalize: bool) -> Encoder:
        """The encoder of the model's own rules, which stays in the cache. It is made and
        pinned under one lock, so no other thread can evict it in between, and nothing is
        pinned when making it fails."""
        return self.get(max_length, normalize, pin=True)

    def get(self, max_length: int | None, normalize: bool, *, pin: bool = False) -> Encoder:
        """The encoder that cuts texts to ``max_length`` tokens (None: not at all) and
        normalizes its rows or not."""
        key = (max_length, normalize)
        with self._lock:
            encoder = self._made.get(key)
            if encoder is None:
                encoder = self._made[key] = self._make(max_length, normalize)
            else:
                self._made.move_to_end(key)
            if pin:
                self._own = key
            self._evict()
            return encoder

    def vocabulary(self) -> Vocabulary:
        """The ``Vocabulary`` that every encoder and lookup of the model shares."""
        with self._lock:
            return self._shared()

    def _shared(self) -> Vocabulary:
        if self._vocabulary is None:
            self._vocabulary = make_vocabulary(self._rules.word_vectors)
        return self._vocabulary

    def _make(self, max_length: int | None, normalize: bool) -> Encoder:
        rules = self._rules
        chars = None if max_length is None else max_length * rules.median_token_length
        spec = TokenizerSpec(
            rules.parameters.identifier, rules.parameters, "drop", max_length, chars
        )
        source: Source | None = rules.source
        content = converted(rules.word_vectors, spec, normalize, source, rules.license)
        return Encoder._new(content, rules.word_vectors, vocabulary=self._shared())

    def _evict(self) -> None:
        """Drop the least recently used encoders beyond the limit, but the model's own."""
        for key in list(self._made):
            if len(self._made) <= self._limit:
                break
            if key != self._own:
                del self._made[key]

    def __len__(self) -> int:
        return len(self._made)
