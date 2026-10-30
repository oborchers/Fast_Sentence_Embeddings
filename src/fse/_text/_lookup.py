#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The lookup of stage 2: the vocabulary table of word vectors, the tokenizer and the
FastText buckets, and the workspace one worker reuses."""

from __future__ import annotations

from dataclasses import dataclass

from fse import _native
from fse._tokenizer_spec import UNIGRAM, TokenizerSpec
from fse._wordvectors import WordVectors

from ._scan import _integer


def tokenize(text: str, tokenizer: str = "fse_words@1") -> list[str]:
    """The tokens of one sentence by a tokenizer, as the native text sources split it."""
    return _native.tokenize(text, tokenizer)


@dataclass(frozen=True, eq=False)
class Vocabulary:
    """The VocabTable of one vocabulary, built once and shared by every lookup of it."""

    table: _native.VocabTable
    fingerprint: str


def _check_word_vectors(word_vectors: object) -> WordVectors:
    if not isinstance(word_vectors, WordVectors):
        raise TypeError(f"word_vectors must be WordVectors, not {type(word_vectors).__name__}")
    return word_vectors


def make_vocabulary(word_vectors: WordVectors) -> Vocabulary:
    """The VocabTable of the words of ``word_vectors``, for ``make_lookup``."""
    wv = _check_word_vectors(word_vectors)
    data, offsets, fingerprint = wv._vocabulary_snapshot()
    return Vocabulary(_native.VocabTable(data, offsets), fingerprint)


def make_lookup(
    word_vectors: WordVectors,
    tokenizer: TokenizerSpec | str,
    *,
    max_ngrams: int | None = None,
    vocabulary: Vocabulary | None = None,
) -> _native.TextLookup:
    """The lookup of stage 2 for word vectors: their vocabulary and FastText buckets.

    ``tokenizer`` is a TokenizerSpec, or the name of a tokenizer without parameters.
    FastText vectors need ``max_ngrams``, the number of ngrams an unknown word uses at
    most (1 to ``MAX_NGRAMS_LIMIT``); other vectors take none. ``vocabulary`` reuses the
    table of ``make_vocabulary``; it must be one of the same words in the same order.
    """
    _check_word_vectors(word_vectors)
    spec = TokenizerSpec.of(tokenizer)
    tokenizer = spec.name
    if vocabulary is None:
        vocabulary = make_vocabulary(word_vectors)
    elif not isinstance(vocabulary, Vocabulary):
        raise TypeError(f"vocabulary must be a Vocabulary, not {type(vocabulary).__name__}")
    elif vocabulary.fingerprint != word_vectors.vocabulary_fingerprint:
        raise ValueError(
            f"the vocabulary {vocabulary.fingerprint} is not the one of the word vectors "
            f"({word_vectors.vocabulary_fingerprint})"
        )
    table = vocabulary.table
    if spec.config is not None:
        return _wordpiece_lookup(word_vectors, spec, table, max_ngrams)
    fasttext = word_vectors.fasttext
    if fasttext is None:
        if max_ngrams is not None:
            raise ValueError("max_ngrams applies only to FastText vectors")
        return _native.TextLookup(table, tokenizer)
    count = _integer(max_ngrams)
    if count is None:
        raise TypeError("FastText vectors need max_ngrams, an int")
    max_ngrams = count
    if not 1 <= max_ngrams <= _native.MAX_NGRAMS_LIMIT:
        raise ValueError(
            f"max_ngrams must be between 1 and {_native.MAX_NGRAMS_LIMIT}, not {max_ngrams}"
        )
    params = (fasttext.min_n, fasttext.max_n, fasttext.bucket)
    return _native.TextLookup(table, tokenizer, params, max_ngrams)


def _wordpiece_lookup(
    word_vectors: WordVectors,
    spec: TokenizerSpec,
    table: _native.VocabTable,
    max_ngrams: int | None,
) -> _native.TextLookup:
    """The lookup of a tokenizer with parameters (wordpiece@1, unigram@1): its state on
    the table of the tokens, token id = row."""
    if spec.config is None or word_vectors.fasttext is not None or max_ngrams is not None:
        raise ValueError(f"{spec.name} needs its parameters and takes no FastText")
    data, offsets, _ = word_vectors._vocabulary_snapshot()
    spec.check_vocabulary(data, offsets)
    arguments = (table, data, offsets, *spec.native_arguments())
    if spec.name == UNIGRAM:
        return _native.TextLookup(table, spec.name, unigram=_native.UnigramState(*arguments))
    return _native.TextLookup(table, spec.name, wordpiece=_native.WordPieceState(*arguments))


MAX_CHUNK_ROWS = _native.MAX_CHUNK_ROWS
MAX_NGRAMS_LIMIT = _native.MAX_NGRAMS_LIMIT
TextLookup = _native.TextLookup
Workspace = _native.Workspace


def new_workspace() -> _native.Workspace:
    """The buffers one worker thread reuses for every chunk it encodes."""
    return _native.Workspace()
