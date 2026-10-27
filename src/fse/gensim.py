#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Convert gensim word vectors into fse ``WordVectors`` (install with ``fse[gensim]``).

gensim is imported only when ``from_gensim`` runs, so fse itself works without it.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from fse import _codes
from fse._validation import validate_arguments
from fse._wordvectors import FastTextParams, WordVectors, _encode_words


def _gensim():
    try:
        import gensim  # noqa: PLC0415 - optional dependency
        from gensim.models.fasttext import FastTextKeyedVectors  # noqa: PLC0415
        from gensim.models.keyedvectors import KeyedVectors  # noqa: PLC0415
    except ImportError as err:
        raise ImportError(
            "fse.gensim needs gensim; install it with: pip install 'fse[gensim]'"
        ) from err
    return gensim, KeyedVectors, FastTextKeyedVectors


@validate_arguments
def from_gensim(
    model: Any, dtype: Literal["float32", "float16", "int8"] = "float32", counts: bool = True
) -> WordVectors:
    """The word vectors of a gensim model as fse ``WordVectors``.

    ``model`` is a ``KeyedVectors`` or ``FastTextKeyedVectors``, or a ``Word2Vec`` or
    ``FastText`` model whose ``wv`` holds them. The vectors are stored as ``dtype``; float16
    and int8 codes are quantized directly from gensim's rows, in blocks, without a float32
    copy. ``counts`` copies gensim's word counts if it has them. For FastText, the rows
    of the words themselves (``vectors_vocab``) and the ngram rows are stored, not
    gensim's combined ``vectors``.
    """
    _, keyed_vectors, fasttext_vectors = _gensim()
    kv = getattr(model, "wv", model)
    if not isinstance(kv, keyed_vectors):
        raise TypeError(
            f"model must be gensim KeyedVectors or a model with them in wv, not {type(model).__name__}"
        )
    words = kv.index_to_key
    if isinstance(kv, fasttext_vectors):
        if not getattr(kv, "compatible_hash", True):
            raise ValueError(
                "FastText vectors with the old, incompatible ngram hash are not supported"
            )
        if not kv.bucket:
            raise ValueError(
                "FastText vectors without ngram buckets; save them as plain KeyedVectors"
            )
        fasttext = FastTextParams(int(kv.min_n), int(kv.max_n), int(kv.bucket))
        rows, ngram_rows = kv.vectors_vocab, kv.vectors_ngrams
    else:
        fasttext, rows, ngram_rows = None, kv.vectors, None
    if rows.dtype != np.float32:
        raise TypeError(f"the word vectors must be float32, not {rows.dtype}")
    if rows.ndim != 2 or rows.shape[0] != len(words):
        raise ValueError(f"gensim holds {len(words)} words but vectors of shape {rows.shape}")
    vocab_bytes, vocab_offsets = _encode_words(words)
    kwargs: dict[str, Any] = {}
    if counts and "count" in getattr(kv, "expandos", {}):
        values = np.asarray(kv.expandos["count"])
        if values.dtype.kind not in "iu":
            raise TypeError(f"gensim's counts must be integers, not {values.dtype}")
        kwargs["counts"] = values.astype(np.int64)
    if fasttext is not None:
        kwargs["fasttext"] = fasttext
    vectors, scales = _stored(rows, dtype, "word vector")
    if ngram_rows is not None:
        kwargs["ngrams"], ngram_scales = _stored(ngram_rows, dtype, "ngram vector")
        if ngram_scales is not None:
            kwargs["ngram_scales"] = ngram_scales
    if scales is not None:
        kwargs["scales"] = scales
    # Codes and vocabulary are new arrays of this call; gensim's float32 rows are copied.
    return WordVectors.from_arrays(
        vectors,
        vocab_bytes=vocab_bytes,
        vocab_offsets=vocab_offsets,
        _copy=dtype == "float32",
        **kwargs,
    )


def _stored(rows: np.ndarray, dtype: str, kind: str):
    if dtype == "float32":
        return np.asarray(rows), None
    return _codes.quantize(rows, dtype, kind=kind)
