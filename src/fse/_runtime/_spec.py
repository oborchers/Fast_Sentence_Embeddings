#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""What an encoder hands to the runtime: its table, weights, steps and tokenizer.

PR 5's Encoder builds one; the runtime prepares a backend from it once. EncodeSpec owns
a frozen copy of the weights, so changing the caller's array afterwards changes nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from fse._backends import PoolInput
from fse._contract import Aggregation, Step, check_steps
from fse._text import MAX_NGRAMS_LIMIT
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors

_FSE_WORDS = TokenizerSpec("fse_words@1")


@dataclass(frozen=True, eq=False)
class EncodeSpec:
    """The stages 1 to 4 of an encoder: tokenizer, word vectors and weights, steps.

    ``weights`` holds one finite float32 weight per word; FastText vectors also need
    ``oov_weight``, the weight of an unknown word, and ``max_ngrams``, the ngrams it uses
    at most. ``steps`` apply to the aggregated rows. ``tokenizer`` is a TokenizerSpec (a
    name stands for one without parameters); its ids must be tokens of the vocabulary.
    """

    word_vectors: WordVectors
    weights: ndarray
    steps: Sequence[Step] = ()
    tokenizer: TokenizerSpec = _FSE_WORDS
    oov_weight: Any = None  # a finite number, kept as given
    max_ngrams: int | None = None
    aggregation: Aggregation = Aggregation.WEIGHTED_MEAN

    def __post_init__(self):
        wv = self.word_vectors
        if not isinstance(wv, WordVectors):
            raise TypeError(f"word_vectors must be WordVectors, not {type(wv).__name__}")
        object.__setattr__(self, "weights", _weights(self.weights, len(wv)))
        tokenizer = TokenizerSpec.of(self.tokenizer)
        object.__setattr__(self, "tokenizer", tokenizer)
        if tokenizer.config is not None:
            if wv.fasttext is not None:
                raise ValueError(f"{tokenizer.name} takes no FastText vectors")
            tokenizer.check_vocabulary(wv.vocab_bytes, wv.vocab_offsets)
        if not isinstance(self.aggregation, Aggregation):
            raise TypeError("aggregation must be an Aggregation")
        dim = self.aggregation.output_dim(wv.dim)
        object.__setattr__(self, "steps", check_steps(self.steps, dim))
        fasttext = wv.fasttext is not None
        if fasttext != (self.oov_weight is not None):
            raise ValueError("oov_weight is given exactly for FastText vectors")
        if fasttext:
            object.__setattr__(self, "oov_weight", _finite(self.oov_weight, "oov_weight"))
        if fasttext != (self.max_ngrams is not None):
            raise ValueError("max_ngrams is given exactly for FastText vectors")
        if fasttext:
            object.__setattr__(self, "max_ngrams", _max_ngrams(self.max_ngrams))

    @property
    def output_dim(self) -> int:
        """The dimension of the finished rows: that of the aggregated rows, as normalize
        keeps it."""
        return self.aggregation.output_dim(self.word_vectors.dim)

    def pool_input(self) -> PoolInput:
        wv = self.word_vectors
        return PoolInput(
            wv.stored_vectors,
            self.weights,
            wv.scales,
            wv.stored_ngrams,
            wv.ngram_scales,
            self.oov_weight,
            self.aggregation,
        )


def _weights(value: object, words: int) -> ndarray:
    if not isinstance(value, ndarray) or value.dtype != np.float32 or value.shape != (words,):
        raise ValueError(f"weights must be a float32 vector of {words} values, one per word")
    copy = np.array(value, copy=True, order="C")
    if not np.isfinite(copy).all():
        raise ValueError(f"weight {int(np.flatnonzero(~np.isfinite(copy))[0])} is not finite")
    copy.setflags(write=False)
    return copy


def _finite(value: object, name: str) -> Any:
    """``value`` itself if it is a finite number. It is not converted: the kernel converts
    it to float32 in its own floating-point environment, where a subnormal stays one."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{name} must be a number, not {type(value).__name__}")
    if not np.isfinite(value):  # flush-to-zero turns no infinity or NaN into a number
        raise ValueError(f"{name} must be finite, not {value}")
    return value


def _max_ngrams(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"max_ngrams must be an int, not {type(value).__name__}")
    if not 1 <= value <= MAX_NGRAMS_LIMIT:
        raise ValueError(f"max_ngrams must be between 1 and {MAX_NGRAMS_LIMIT}, not {value}")
    return int(value)
