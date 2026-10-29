"""Tiny Model2Vec models written by model2vec itself, and the bound of a comparison.

fse and model2vec both pool float32 rows of the same token ids; fse sums by fma in token
order and scales by ``1/n``, model2vec takes numpy's ``mean`` (pairwise sums, then a
division) and divides by ``norm + 1e-32``. So rows agree within the bound of the tolerance
contract (ARCHITECTURE.md), computed here from the ids and the table: before normalize
``(TAU + 2 gamma_K) m + DELTA K sqrt(d)``, after it ``2 E / r`` plus the rounding of both
normalizations, with K the ids of a text, m the norm of ``(1/K) sum |x|`` and r the norm of
fse's row before normalize.
"""

import numpy as np

from fse import _native, _reference
from fse._text import StringSource
from support import wordpiece as wp
from support.bound import DELTA, TAU, U, gamma

DIM = 16


def rows(n_tokens: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_tokens, DIM)).astype(np.float32)


def save_model(folder, table=None, *, normalize=True, max_length=512, tokens=None):
    """A Model2Vec model saved by model2vec 0.10.0 into ``folder``."""
    from model2vec import StaticModel

    tokens = wp.vocabulary() if tokens is None else tokens
    tokenizer = wp.hf_tokenizer(tokens, {t: tokens.index(t) for t in wp.SPECIAL if t in tokens})
    table = rows(len(tokens)) if table is None else table
    StaticModel(table, tokenizer, normalize=normalize, max_length=max_length).save_pretrained(
        folder
    )
    return folder


def ids_of(encoder, texts):
    """The ids fse pools for every text."""
    lookup = encoder._pipeline("cpu").lookup
    source = StringSource(texts)
    batches = [source.encode(c, lookup, _native.Workspace()) for c in source.scan().chunks]
    return [
        b.tokens[b.offsets[s] : b.offsets[s + 1]] for b in batches for s in range(b.n_sentences)
    ]


def assert_close_to_model2vec(got, want, encoder, texts, normalize):
    """``got`` (fse) and ``want`` (model2vec) agree within the bound, row by row; empty
    texts are zero on both sides. Gives the largest ratio of error to bound."""
    table = np.asarray(encoder.word_vectors.stored_vectors, np.float32)
    assert got.shape == want.shape == (len(texts), table.shape[1])
    worst = 0.0
    for i, ids in enumerate(ids_of(encoder, texts)):
        k = len(ids)
        if k == 0:
            assert not got[i].any(), i
            assert not want[i].any(), i
            continue
        m = np.linalg.norm(np.abs(table[ids].astype(np.float64)).sum(axis=0) / k)
        bound = (TAU + 2 * gamma(k)) * m + DELTA * k * np.sqrt(table.shape[1])
        if normalize:
            batch = _reference_batch(ids)
            r = np.linalg.norm(
                _reference.pool_weighted_mean(batch, table, np.ones(len(table), np.float32))[
                    0
                ].astype(np.float64)
            )
            bound = 2 * bound / r + 2 * (gamma(table.shape[1]) / 2 + 4 * U)
        error = np.linalg.norm(got[i].astype(np.float64) - want[i].astype(np.float64))
        assert error <= bound, f"text {i} ({texts[i][:40]!r}): {error} > {bound}"
        worst = max(worst, error / bound)
    return worst


def _reference_batch(ids):
    from fse._contract import Batch

    return Batch(0, np.array([0, len(ids)], np.int64), np.asarray(ids, np.uint32))
