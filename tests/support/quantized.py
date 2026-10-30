"""The oracle of a quantized encoder: the numerical contract applied by the reference to the
table the reference quantized, independent of ``WordVectors.quantize`` and of the kernels.

The ids of a batch do not depend on the dtype of the table, so a test passes the batch of
the float32 encoder (stages 1 and 2) and gets the rows the quantized encoder must give,
bit for bit.
"""

import numpy as np

from fse import _reference, _reference_text
from fse import _reference_head as ref

F32 = np.float32


def tables(wv, dtype):
    """``(vectors, scales, ngrams, ngram_scales)`` of the float32 tables of ``wv`` after
    ``_reference.quantize_rows``; the ngram table is quantized on its own."""
    codes, scales = _reference.quantize_rows(wv.stored_vectors, dtype)
    if wv.stored_ngrams is None:
        return codes, scales, None, None
    grams, gram_scales = _reference.quantize_rows(wv.stored_ngrams, dtype)
    return codes, scales, grams, gram_scales


def rows(batch, wv, dtype, steps=(), oov_weight=None, first_row=0):
    """The rows of ``batch`` for the float32 word vectors ``wv`` stored as ``dtype``: every
    word weighs 1, then the steps."""
    codes, scales, grams, gram_scales = tables(wv, dtype)
    out = _reference.pool_weighted_mean(
        batch, codes, np.ones(len(wv), F32), scales, grams, gram_scales, oov_weight
    )
    return _reference.postprocess(out, steps, first_row) if steps else out


def head_logits(batch, wv, dtype, weight, bias, steps=(), oov_weight=None, folded=True):
    """The logits of the head (``weight``, ``bias``) on the rows of ``batch`` for ``wv``
    stored as ``dtype``: A folded into the quantized rows (a head on raw means), or applied
    to the pooled rows after ``steps``."""
    if not folded:
        return ref.logits(rows(batch, wv, dtype, steps, oov_weight), weight, bias)
    codes, scales, grams, gram_scales = tables(wv, dtype)
    folded_grams = None if grams is None else ref.fold(grams, gram_scales, weight)
    pooled = _reference.pool_weighted_mean(
        batch,
        ref.fold(codes, scales, weight),
        np.ones(len(wv), F32),
        None,
        folded_grams,
        None,
        oov_weight,
    )
    return ref.logits(pooled, None, bias)


def words_batch(wv, sentences, tokenizer="fse_words@1", max_ngrams=0):
    """The reference batch of ``sentences`` for the words of ``wv`` (FastText unknown words
    take their first ``max_ngrams`` buckets)."""
    index = {wv.word(i).encode(): i for i in range(len(wv))}
    fasttext = None if wv.fasttext is None else tuple(wv.fasttext)
    return _reference_text.encode_texts(
        sentences, index, tokenizer=tokenizer, fasttext=fasttext, max_ngrams=max_ngrams
    )


def bits(rows):
    """The bit patterns of float32 rows, to compare bits and not values."""
    return np.ascontiguousarray(rows, dtype=F32).view(np.uint32).tobytes()
