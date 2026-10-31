#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The fse encoders a ``StaticModel`` can be (docs/architecture/hub.md, "model2vec compatibility (fse.StaticModel)").

A ``StaticModel`` encodes the way model2vec does: it pools the float32 rows of the token ids
with the unknown id dropped, cuts a text to ``max_length`` tokens of its first
``max_length * median token length`` characters, and normalizes or not. An fse encoder is
one only if it fits that closed profile; everything else (a tokenizer without parameters,
an unknown id that is kept, another length rule, a step model2vec has no name for) is
refused with the reason, as its rows would not be model2vec's.
"""

from __future__ import annotations

from dataclasses import replace

from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._format import Source
from fse._encoder._model2vec import median_length
from fse._encoder._model2vec_rules import Rules, Unset, reduced
from fse._wordvectors import WordVectors

NORMALIZE = "normalize@1"


def profile_rules(
    encoder: Encoder,
    *,
    normalize: bool | None,
    max_length: int | Unset | None,
    dimensionality: int | None,
) -> Rules:
    """The ``Rules`` of an fse ``encoder`` that fits the profile: its own ``normalize`` and
    length rule, unless ``normalize`` or ``max_length`` (not ``UNSET``) state others, and
    its first ``dimensionality`` columns; ``base_model_name`` is the Hub repository of its
    ``fse.source``. Anything else raises UnsupportedArtifactError."""
    content = encoder._content
    spec, words = content.tokenizer, encoder.word_vectors
    reason = _misfit(encoder)
    tokens: list[str] = []
    if reason is None:
        tokens = [words.word(i) for i in range(len(words))]
        median = median_length(tokens)
        if spec.max_chars != (None if spec.max_tokens is None else spec.max_tokens * median):
            reason = f"cuts texts to {spec.max_tokens} tokens of {spec.max_chars} characters"
    if reason is not None or spec.config is None:
        raise UnsupportedArtifactError(
            f"the encoder {reason}, which fse.StaticModel does not offer: it encodes as "
            "model2vec does (float32 rows, a WordPiece or Unigram tokenizer, the unknown id "
            "dropped, model2vec's length rule, optionally normalized); use fse.Encoder for "
            "this model"
        )
    words, source = _first_columns(words, tokens, content.source, dimensionality)
    return Rules(
        words,
        tuple(tokens),
        spec.config,
        {},
        (NORMALIZE in encoder.steps) if normalize is None else normalize,
        spec.max_tokens if isinstance(max_length, Unset) else max_length,
        median,
        None if source is None else source.repo,
        None,
        source,
        content.license,
    )


def _misfit(encoder: Encoder) -> str | None:
    """Why the encoder is no model2vec model (judged without its vocabulary), or None."""
    content = encoder._content
    spec, words = content.tokenizer, encoder.word_vectors
    if spec.config is None:
        return f"has the tokenizer {spec.name}, which has no parameters"
    if encoder.steps not in ((), (NORMALIZE,)):
        return f"has the steps {list(encoder.steps)}"
    if spec.unknown != "drop":
        return f"keeps the unknown id ({spec.name})"
    if words.dtype != "float32":
        return f"has a {words.dtype} table (model2vec computes the mean in that dtype)"
    if words.fasttext is not None or content.oov_weight is not None:
        return "has FastText vectors"
    return None


def _first_columns(
    words: WordVectors, tokens: list[str], source: Source | None, dimensionality: int | None
) -> tuple[WordVectors, Source | None]:
    """The table cut to its first ``dimensionality`` columns, and its provenance with the
    width of the model it came from."""
    table, kept = reduced(words.stored_vectors, dimensionality)
    if kept is None:
        return words, source
    cut = WordVectors._from_arrays(table, tokens, None, None, None, {}, copy=False)
    if source is None:
        return cut, None
    first = words.dim if source.dims is None else source.dims[0]
    return cut, replace(source, dims=(first, kept))
