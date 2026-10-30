#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The token ids of texts under a length rule (ARCHITECTURE.md, "model2vec compatibility").

``tokenize``, ``encode_as_sequence`` and the tokenizer facade of ``StaticModel`` need the
ids a text is pooled from, which ``Encoder`` never hands out. They come from the lookup of
stage 2 and the text source of stage 1 (``fse._text``), the code that ``encode`` runs, so
ids and rows cannot disagree. A length rule is a pair of cuts (tokens, characters) and
what to do with the unknown id; each distinct rule is a lookup of its own on the shared
``Vocabulary``, made on first use and kept in a bounded cache.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from numpy import ndarray

from fse._encoder._model2vec_rules import Rules
from fse._text import StringSource, TextLookup, Vocabulary, make_lookup, new_workspace
from fse._tokenizer_spec import TokenizerSpec

LIMIT = 8
Rule = tuple[int | None, int | None, str]


class Ids:
    """The ids of the texts of one model, by length rule ``(max_tokens, max_chars,
    unknown)`` (None: not cut; unknown ``"drop"`` or ``"keep"``). Several threads may ask
    at once; each call has its own workspace."""

    def __init__(self, rules: Rules, vocabulary: Callable[[], Vocabulary], limit: int = LIMIT):
        self._rules = rules
        self._vocabulary = vocabulary
        self._limit = limit
        self._lock = threading.Lock()
        self._made: OrderedDict[Rule, TextLookup] = OrderedDict()

    def lookup(self, rule: Rule) -> TextLookup:
        """The lookup of ``rule``."""
        with self._lock:
            lookup = self._made.get(rule)
            if lookup is None:
                lookup = self._made[rule] = self._make(rule)
                while len(self._made) > self._limit:
                    self._made.popitem(last=False)
            else:
                self._made.move_to_end(rule)
            return lookup

    def _make(self, rule: Rule) -> TextLookup:
        max_tokens, max_chars, unknown = rule
        parameters = self._rules.parameters
        spec = TokenizerSpec(parameters.identifier, parameters, unknown, max_tokens, max_chars)
        return make_lookup(self._rules.word_vectors, spec, vocabulary=self._vocabulary())

    def chunks(self, texts: Any, rule: Rule) -> Iterator[list[ndarray]]:
        """The ids of ``texts`` (a list, tuple or numpy array of str), one uint32 array per
        text, a list for each chunk of the source. The arrays are views of buffers that the
        next chunk reuses: use them before asking for it."""
        lookup = self.lookup(rule)
        source = StringSource(texts)
        workspace = new_workspace()
        for chunk in source.scan().chunks:
            batch = source.encode(chunk, lookup, workspace)
            tokens, offsets = batch.tokens, batch.offsets
            yield [
                np.asarray(tokens[offsets[i] : offsets[i + 1]]) for i in range(batch.n_sentences)
            ]

    def lists(self, texts: Any, rule: Rule) -> list[list[int]]:
        """The ids of ``texts`` as lists of ints, one list per text."""
        return [row.tolist() for rows in self.chunks(texts, rule) for row in rows]

    def __len__(self) -> int:
        return len(self._made)
