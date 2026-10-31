"""The ids a converted encoder pools, and the bound of its rows against the original
library (model2vec, sentence-transformers).

fse and the original both pool float32 rows of the same token ids; fse sums by fma in token
order and scales by ``1/n``, model2vec takes numpy's ``mean`` (pairwise sums, then a
division), sentence-transformers' EmbeddingBag a sum in token order divided by n. So rows
agree within the conversion parity bound, computed here from the
ids and the table: before normalize ``(TAU + 2 gamma_K) m + DELTA K sqrt(d)``, after it
``2 E / r`` plus the rounding of both normalizations, with K the ids of a text, m the norm
of ``(1/K) sum |x|`` and r the norm of fse's row before normalize. A row that is zero
before normalize (an empty text, a zero table, rows that cancel exactly) has no such bound
(``r = 0``): it must be zero on both sides.
"""

import numpy as np

from fse import _native, _reference
from fse._contract import Batch
from fse._text import StringSource

TAU = 1e-5
DELTA = 2.0**-125
U = 2.0**-24


def gamma(k):
    return k * U / (1 - k * U)


def ids_of(encoder, texts):
    """The ids fse pools for every text."""
    lookup = encoder._pipeline("cpu").lookup
    source = StringSource(texts)
    batches = [source.encode(c, lookup, _native.Workspace()) for c in source.scan().chunks]
    return [
        b.tokens[b.offsets[s] : b.offsets[s + 1]] for b in batches for s in range(b.n_sentences)
    ]


def assert_close(got, want, encoder, texts, normalize):
    """``got`` (fse) and ``want`` (the original) agree within the bound, row by row; rows
    that are zero before normalize are zero on both sides. Gives the largest ratio of
    error to bound."""
    table = np.asarray(encoder.word_vectors.stored_vectors, np.float32)
    assert got.shape == want.shape == (len(texts), table.shape[1])
    worst = 0.0
    for i, ids in enumerate(ids_of(encoder, texts)):
        k = len(ids)
        raw = _raw(ids, table)
        if not raw.any():
            assert not got[i].any(), i
            assert not want[i].any(), i
            continue
        m = np.linalg.norm(np.abs(table[ids].astype(np.float64)).sum(axis=0) / k)
        bound = (TAU + 2 * gamma(k)) * m + DELTA * k * np.sqrt(table.shape[1])
        if normalize:
            r = np.linalg.norm(raw.astype(np.float64))
            bound = 2 * bound / r + 2 * (gamma(table.shape[1]) / 2 + 4 * U)
        error = np.linalg.norm(got[i].astype(np.float64) - want[i].astype(np.float64))
        assert error <= bound, f"text {i} ({texts[i][:40]!r}): {error} > {bound}"
        worst = max(worst, error / bound)
    return worst


def _raw(ids, table):
    """fse's row before normalize (the reference's bits)."""
    batch = Batch(0, np.array([0, len(ids)], np.int64), np.asarray(ids, np.uint32))
    return _reference.pool_weighted_mean(batch, table, np.ones(len(table), np.float32))[0]
