#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""fit: from word vectors and a corpus to a frozen encoder (ARCHITECTURE.md, "2.0 design").

The passes over the corpus are planned before it is read: a count of its tokens for the
sentence length of uSIF (unless given), then one pass of the weighted rows into a
GramSink for the components and the projection. A corpus that can be read only once (a
stream) raises before it is read if it would need more than one pass. The learned linear
steps are folded into the word vectors by default; ``normalize`` stays a step per row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _gram
from fse._contract import Normalize, Project, RemoveComponents, Step
from fse._runtime import EncodeSpec, GramSink, Pipeline, count_tokens
from fse._text import Vocabulary, make_vocabulary
from fse._wordvectors import WordVectors

from ._encoder import Encoder
from ._fold import fold
from ._format import EncoderContent
from ._sources import is_stream, opened
from ._weights import check_counts, sif_weights, usif_component_weights, usif_weights

_F32 = np.float32
DEFAULT_MAX_NGRAMS = 40


@dataclass(frozen=True)
class FitSpec:
    """What a recipe learns: its word weights, components, projection and steps.

    ``weighting`` is ``none`` (Average), ``sif`` or ``usif``; ``length`` None for uSIF
    learns the average sentence length from the corpus.
    """

    name: str
    weighting: str
    components: int
    dim: int | None
    normalize: bool
    fold: bool
    tokenizer: str
    max_ngrams: int | None
    oov_weight: float | None
    alpha: float = 1e-3
    length: int | None = None

    @property
    def counts_pass(self) -> bool:
        return self.weighting == "usif" and self.length is None

    @property
    def gram_pass(self) -> bool:
        return self.components > 0 or self.dim is not None

    def passes(self) -> int:
        return int(self.counts_pass) + int(self.gram_pass)


def fit(
    spec: FitSpec, word_vectors: WordVectors, texts: Any, *, counts: Any, workers: Any, backend: str
) -> Encoder:
    """The encoder of ``spec`` fitted on ``texts`` (None when it learns nothing from them)."""
    return _fit(spec, word_vectors, texts, counts, workers, backend, None)[0]


def fit_transform(
    spec: FitSpec,
    word_vectors: WordVectors,
    texts: Any,
    *,
    counts: Any,
    workers: Any,
    backend: str,
    dtype: str,
) -> tuple[Encoder, Any]:
    """The fitted encoder and the rows of ``texts`` it gives, as ``encode`` gives them:
    the source is read once more after the fit (ARCHITECTURE.md, "2.0 design")."""
    return _fit(spec, word_vectors, texts, counts, workers, backend, dtype)


def _fit(spec, word_vectors, texts, counts, workers, backend, dtype) -> tuple[Encoder, Any]:
    if not isinstance(word_vectors, WordVectors):
        raise TypeError(f"word_vectors must be WordVectors, not {type(word_vectors).__name__}")
    passes = spec.passes() + (dtype is not None)
    if passes == 0 and texts is not None:
        raise ValueError(f"{spec.name} learns nothing from texts with these parameters; pass none")
    if passes and texts is None:
        raise ValueError(f"{spec.name} learns from a corpus with these parameters: pass texts")
    counts = _counts(spec, word_vectors, counts)
    vocabulary = make_vocabulary(word_vectors)
    if texts is None:
        weights = _weights(spec, counts, spec.length)
        steps = _normalize(spec)
        return _encoder(spec, word_vectors, vocabulary, weights, steps, 0, spec.length), None
    with opened(texts, spec.tokenizer) as source:
        if passes > 1 and is_stream(source):
            raise ValueError(
                f"{spec.name} reads the corpus {passes} times with these parameters, but a "
                "stream is read once: pass a list or a file (or length= for uSIF)"
            )
        length = spec.length
        if spec.counts_pass:
            length = _learned_length(spec, word_vectors, vocabulary, source, workers, backend)
        weights = _weights(spec, counts, length)
        steps, sentences = _learn(spec, word_vectors, vocabulary, weights, source, workers, backend)
        encoder = _encoder(spec, word_vectors, vocabulary, weights, steps, sentences, length)
        if dtype is None:
            return encoder, None
        return encoder, encoder.encode(source, dtype=dtype, workers=workers, backend=backend)


def _counts(spec: FitSpec, word_vectors: WordVectors, counts: Any) -> ndarray | None:
    """The checked counts of the words, for SIF and uSIF; Average takes none."""
    if spec.weighting == "none":
        if counts is not None:
            raise ValueError("Average weights every word by 1 and takes no counts")
        return None
    if counts is None:
        counts = word_vectors.counts
        if counts is None:
            raise ValueError(
                f"{spec.name} weights words by their frequency, and these word vectors have no "
                "counts: pass counts=, one per word (for example from wordfreq)"
            )
    return check_counts(counts, len(word_vectors))


def _weights(spec: FitSpec, counts: ndarray | None, length: int | None) -> ndarray | None:
    if counts is None:
        return None
    if spec.weighting == "sif":
        return sif_weights(counts, spec.alpha)
    assert length is not None
    return usif_weights(counts, length)


def _pipeline(spec, word_vectors, vocabulary, weights, backend) -> Pipeline:
    weights = np.ones(len(word_vectors), _F32) if weights is None else weights
    fasttext = word_vectors.fasttext is not None
    return Pipeline(
        EncodeSpec(
            word_vectors,
            weights,
            (),
            spec.tokenizer,
            oov_weight=_oov_weight(spec, weights) if fasttext else None,
            max_ngrams=(spec.max_ngrams or DEFAULT_MAX_NGRAMS) if fasttext else None,
        ),
        backend,
        vocabulary=vocabulary,
    )


def _oov_weight(spec: FitSpec, weights: ndarray) -> Any:
    """The weight of an unknown FastText word: given, 1 for Average, else the largest
    word weight (as 1.x)."""
    if spec.oov_weight is not None:
        return _F32(spec.oov_weight)
    return _F32(1) if spec.weighting == "none" else _F32(weights.max())


def _learned_length(spec, word_vectors, vocabulary, source, workers, backend) -> int:
    pipeline = _pipeline(spec, word_vectors, vocabulary, None, backend)
    sentences, tokens = count_tokens(pipeline, source, workers=workers)
    if not tokens:
        raise ValueError(
            f"uSIF learns the sentence length from the corpus, which has {sentences} "
            "sentences and no tokens"
        )
    return max(1, tokens // sentences)


def _learn(spec, word_vectors, vocabulary, weights, source, workers, backend):
    """The learned steps and the number of sentences they were learned on."""
    if not spec.gram_pass:
        return _normalize(spec), 0
    pipeline = _pipeline(spec, word_vectors, vocabulary, weights, backend)
    dim = pipeline.output_dim
    sink = GramSink(dim, pipeline.backend)
    pipeline.infer(source, workers=workers, sink=sink)
    gram, sentences = sink.result(), sink.next_row
    if not np.trace(gram) > 0:
        raise ValueError(
            f"the corpus of {sentences} sentences gives no sentence vector that is not zero "
            f"(no known word); {spec.name} learns nothing from it"
        )
    steps: list[Step] = []
    removed = component_weights = None
    if spec.components:
        if spec.components > min(sentences, dim):
            raise ValueError(
                f"{spec.name} removes {spec.components} components, which needs at least as "
                f"many sentences and dimensions; the corpus has {sentences} sentences"
            )
        singular_values, removed = _gram.components_from_gram(
            gram, sentences, spec.components, _F32
        )
        if spec.weighting == "usif":
            component_weights = usif_component_weights(singular_values)
        steps.append(RemoveComponents(removed, component_weights))
    if spec.dim is not None:
        basis = _gram.projection_from_gram(
            gram, sentences, spec.dim, _F32, removed, component_weights
        )
        steps.append(Project(basis))
    return (*steps, *_normalize(spec)), sentences


def _normalize(spec: FitSpec) -> tuple[Step, ...]:
    return (Normalize(),) if spec.normalize else ()


def _encoder(spec, word_vectors, vocabulary: Vocabulary, weights, steps, sentences, length):
    linear = tuple(s for s in steps if isinstance(s, RemoveComponents | Project))
    kept = tuple(s for s in steps if not isinstance(s, RemoveComponents | Project))
    folded = spec.fold and bool(linear)
    pooled = fold(word_vectors, linear) if folded else word_vectors
    fasttext = word_vectors.fasttext is not None
    w = np.ones(len(word_vectors), _F32) if weights is None else weights
    content = EncoderContent(
        tokenizer=spec.tokenizer,
        steps=kept if folded else tuple(steps),
        input_dim=pooled.dim,
        weights=weights,
        oov_weight=np.array(_oov_weight(spec, w), _F32) if fasttext else None,
        max_ngrams=(spec.max_ngrams or DEFAULT_MAX_NGRAMS) if fasttext else None,
        word_vectors=pooled.fingerprint,
        vocabulary=pooled.vocabulary_fingerprint,
        recipe=_recipe(spec, sentences, length),
        folded=linear if folded else (),
        folded_from=word_vectors.fingerprint if folded else None,
    )
    return Encoder._new(content, pooled, vocabulary=vocabulary)


def _recipe(spec: FitSpec, sentences: int, length: int | None) -> dict[str, Any]:
    recipe: dict[str, Any] = {
        "name": spec.name.lower(),
        "version": 1,
        "components": spec.components,
        "dim": spec.dim,
        "normalize": spec.normalize,
        "fold": spec.fold,
        "sentences": sentences,
    }
    if spec.weighting == "sif":
        recipe["alpha"] = spec.alpha
    if spec.weighting == "usif":
        recipe["length"] = length
    return recipe
