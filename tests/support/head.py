"""The reference of a linear head on an encoder: what the runtime and the models must give,
bit for bit, composed from fse._reference, fse._reference_text and fse._reference_head."""

import numpy as np

from fse import _reference, _reference_text
from fse import _reference_head as ref
from fse._contract import Normalize

F32 = np.float32


def reference_batch(spec, sentences, first_row=0):
    """The batch of ``sentences`` for an EncodeSpec, by the reference tokenizers."""
    wv = spec.word_vectors
    index = {wv.word(i).encode(): i for i in range(len(wv))}
    fasttext = None if wv.fasttext is None else tuple(wv.fasttext)
    return _reference_text.encode_texts(
        sentences,
        index,
        tokenizer=spec.tokenizer,
        fasttext=fasttext,
        max_ngrams=spec.max_ngrams or 0,
        first_row=first_row,
    )


def folded_pool(pool, weight):
    """The arrays of the folded table: (vectors, weights, scales, ngrams, ngram scales,
    oov_weight) as pool_weighted_mean takes them."""
    ngrams = (
        None
        if pool.ngram_vectors is None
        else ref.fold(pool.ngram_vectors, pool.ngram_scales, weight)
    )
    return (
        ref.fold(pool.vectors, pool.scales, weight),
        pool.weights,
        None,
        ngrams,
        None,
        pool.oov_weight,
    )


def pooled(spec, sentences, steps=(), first_row=0):
    """The pooled rows of ``sentences`` with ``steps`` (not the encoder's)."""
    batch = reference_batch(spec, sentences, first_row)
    p = spec.pool_input()
    rows = _reference.pool_weighted_mean(
        batch, p.vectors, p.weights, p.scales, p.ngram_vectors, p.ngram_scales, p.oov_weight
    )
    return _reference.postprocess(rows, steps, first_row) if steps else rows


def logits(spec, weight, bias, sentences, *, normalized=False, folded=None, first_row=0):
    """The logits of a head (``weight`` (k, d), ``bias``) by ``linear@1``: folded for a head
    on raw means unless ``folded`` is False, else on the pooled (normalized) rows."""
    if folded is None:
        folded = not normalized
    if folded:
        if normalized:
            raise ValueError("a normalized head does not fold")
        batch = reference_batch(spec, sentences, first_row)
        rows = _reference.pool_weighted_mean(batch, *folded_pool(spec.pool_input(), weight))
        return ref.logits(rows, None, bias, first_row)
    steps = (Normalize(),) if normalized else ()
    return ref.logits(pooled(spec, sentences, steps, first_row), weight, bias, first_row)


def outputs(z, output, link=None):
    """The outputs of finished logits: themselves, the probabilities of ``link`` or labels."""
    if output == "logits":
        return z
    if output == "labels":
        return ref.labels(z)
    return ref.proba(z, link)
