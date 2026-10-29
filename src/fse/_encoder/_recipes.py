#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The recipes of fse 2.0: Average, SIF and USIF (ARCHITECTURE.md, "2.0 design").

A recipe holds parameters only; ``fit`` learns what it needs from word vectors and a
corpus and returns a frozen ``Encoder``, whose ``encode`` gives the vectors. By default the learned linear steps are folded into the encoder's word vectors
and every row ends normalized.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from fse._contract import check_tokenizer
from fse._text import MAX_NGRAMS_LIMIT
from fse._validation import NonNegativeInt, PositiveFloat, PositiveInt, validate_arguments
from fse._wordvectors import WordVectors

from ._encoder import Encoder
from ._fit import FitSpec, fit

Backend = Literal["cpu", "numpy"]


class _Recipe:
    __slots__ = ("_spec",)
    _spec: FitSpec

    def __setattr__(self, name: str, value: Any):
        if hasattr(self, "_spec"):
            raise AttributeError(f"a {type(self).__name__} recipe does not change; make a new one")
        object.__setattr__(self, name, value)

    @validate_arguments
    def fit(
        self,
        word_vectors: Any,
        texts: Any = None,
        *,
        counts: Any = None,
        workers: PositiveInt | None = None,
        backend: Backend = "cpu",
    ) -> Encoder:
        """The encoder learned from ``word_vectors`` and the corpus ``texts`` (a list of
        str, a path, token lists for the external tokenizer, or an iterable of str), which
        is needed exactly when something is learned from it: components, ``dim`` or the
        uSIF length. ``counts`` (one per word) replace the counts of the word vectors."""
        spec = self._checked(word_vectors)
        return fit(spec, word_vectors, texts, counts=counts, workers=workers, backend=backend)

    def _checked(self, word_vectors: Any) -> FitSpec:
        if not isinstance(word_vectors, WordVectors):
            raise TypeError(f"word_vectors must be WordVectors, not {type(word_vectors).__name__}")
        spec = self._spec
        if word_vectors.fasttext is None and (
            spec.max_ngrams is not None or spec.oov_weight is not None
        ):
            raise ValueError("max_ngrams and oov_weight apply only to FastText vectors")
        return spec

    def __repr__(self) -> str:
        s = self._spec
        return (
            f"{type(self).__name__}(components={s.components}, dim={s.dim}, "
            f"normalize={s.normalize}, fold={s.fold})"
        )

    def __reduce__(self):
        return (_rebuild, (type(self), self._spec))


def _rebuild(cls: type, spec: FitSpec) -> _Recipe:
    out: _Recipe = object.__new__(cls)
    object.__setattr__(out, "_spec", spec)
    return out


def _fit_spec(name: str, weighting: str, **kwargs: Any) -> FitSpec:
    max_ngrams = kwargs["max_ngrams"]
    if max_ngrams is not None and max_ngrams > MAX_NGRAMS_LIMIT:
        raise ValueError(f"max_ngrams must be at most {MAX_NGRAMS_LIMIT}, not {max_ngrams}")
    oov_weight = kwargs["oov_weight"]
    if oov_weight is not None and not np.isfinite(oov_weight):
        raise ValueError("oov_weight must be finite")
    alpha = kwargs.get("alpha")
    with np.errstate(over="ignore"):
        as_float32 = None if alpha is None else np.float32(alpha)
    if as_float32 is not None and not (np.isfinite(as_float32) and as_float32 > 0):
        raise ValueError(f"alpha must be positive and finite in float32, not {alpha!r}")
    kwargs["tokenizer"] = check_tokenizer(kwargs["tokenizer"])
    return FitSpec(name=name, weighting=weighting, **kwargs)


class Average(_Recipe):
    """The plain average of the word vectors of a sentence (every word weighs 1).

    It learns nothing unless ``dim`` asks for a projection to that many dimensions
    (PCA of the averaged rows of a corpus).
    """

    __slots__ = ()

    @validate_arguments
    def __init__(
        self,
        *,
        dim: PositiveInt | None = None,
        normalize: bool = True,
        fold: bool = True,
        tokenizer: str = "fse_words@1",
        max_ngrams: PositiveInt | None = None,
        oov_weight: float | None = None,
    ):
        self._spec = _fit_spec(
            "Average",
            "none",
            components=0,
            dim=dim,
            normalize=normalize,
            fold=fold,
            tokenizer=tokenizer,
            max_ngrams=max_ngrams,
            oov_weight=oov_weight,
        )


class SIF(_Recipe):
    """Smooth inverse frequency (Arora et al., 2017): words weigh alpha / (alpha + p(w)),
    and the first ``components`` principal components of the corpus are removed."""

    __slots__ = ()

    @validate_arguments
    def __init__(
        self,
        *,
        alpha: PositiveFloat = 1e-3,
        components: NonNegativeInt = 1,
        dim: PositiveInt | None = None,
        normalize: bool = True,
        fold: bool = True,
        tokenizer: str = "fse_words@1",
        max_ngrams: PositiveInt | None = None,
        oov_weight: float | None = None,
    ):
        self._spec = _fit_spec(
            "SIF",
            "sif",
            alpha=alpha,
            components=components,
            dim=dim,
            normalize=normalize,
            fold=fold,
            tokenizer=tokenizer,
            max_ngrams=max_ngrams,
            oov_weight=oov_weight,
        )


class USIF(_Recipe):
    """Unsupervised smooth inverse frequency (Ethayarajh, 2018): word weights from the
    vocabulary size and the average sentence ``length`` (learned from the corpus if
    None), and ``components`` principal components removed, each weighted by its share of
    the variance."""

    __slots__ = ()

    @validate_arguments
    def __init__(
        self,
        *,
        length: PositiveInt | None = None,
        components: NonNegativeInt = 5,
        dim: PositiveInt | None = None,
        normalize: bool = True,
        fold: bool = True,
        tokenizer: str = "fse_words@1",
        max_ngrams: PositiveInt | None = None,
        oov_weight: float | None = None,
    ):
        self._spec = _fit_spec(
            "USIF",
            "usif",
            length=length,
            components=components,
            dim=dim,
            normalize=normalize,
            fold=fold,
            tokenizer=tokenizer,
            max_ngrams=max_ngrams,
            oov_weight=oov_weight,
        )
