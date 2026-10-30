"""The error bound of a head's logits against an exact head in float64 (ARCHITECTURE.md,
"Numerical contract", linear@1), per row and output.

The reference value is scikit-learn's: ``z* = A64 r + b64`` in float64 on the float32
vector r that fse gives (``encode``). With u = 2**-24, gamma(m) = m u / (1 - m u) and g =
ceil(d / 16) + 4 (the longest rounding chain of dot16):

- unfolded and normalized paths (fse scores the same r): |z - z*| <= u sum|a64||r| + u|b64|
  (rounding A and b to float32) + gamma(g) sum|a||r| (dot16) + u (|z| + |b|) (the bias add)
  + the float64 error of z*;
- folded path: fse pools the folded rows instead, so the bound also covers the error of r
  itself and of the fold. With c_t the exact contributions of the sentence (factor times
  stored row, n tokens, K contributions), r_exact = (1/n) sum c_t: |r - r_exact| <= E =
  gamma(K + 3) (1/n) sum|c_t|, and the folded logits are within gamma(g + 1) (1/n) sum_t
  sum_i |a_i||c_ti| + gamma(K + 2) (1/n) sum_t |f_t||t_t| + u sum|a64||r_exact| + u|b64| +
  u (|z| + |b|) of A64 r_exact + b64, which is within sum|a64| E of z*.

The bound holds while no intermediate value overflows or underflows float32; the tests
use values far from both limits.
"""

import math

import numpy as np

from fse._contract import OOV_TOKEN

U = 2.0**-24
U64 = 2.0**-53


def gamma(m, u=U):
    return m * u / (1 - m * u)


def chain(d):
    return math.ceil(d / 16) + 4


def _float64_error(a64, r):
    """The error of scikit-learn's float64 product, in any order."""
    return gamma(a64.shape[1] + 1, U64) * (np.abs(r) @ np.abs(a64).T)


def unfolded(r, a64, b64, a, b, z):
    """The bound of fse's logits ``z`` (n, k) on the float32 vectors ``r`` (n, d) against
    scikit-learn's ``A64 r + b64``."""
    r = np.asarray(r, np.float64)
    a64, b64 = np.asarray(a64, np.float64), np.asarray(b64, np.float64)
    a, b, z = (np.asarray(x, np.float64) for x in (a, b, z))
    coefficients = U * (np.abs(r) @ np.abs(a64).T) + U * np.abs(b64)
    dot = gamma(chain(r.shape[1])) * (np.abs(r) @ np.abs(a).T)
    return coefficients + dot + U * (np.abs(z) + np.abs(b)) + _float64_error(a64, r)


def _unknown(pool, buckets):
    """The contributions of an unknown word with ``buckets``: (g, s, y, where) each."""
    if not len(buckets):
        return []
    share = np.float32(np.float32(1) / np.float32(len(buckets)))
    factor = float(np.float32(share * np.float32(pool.oov_weight)))
    scales = pool.ngram_scales
    return [
        (
            factor,
            1.0 if scales is None else float(scales[b]),
            pool.ngram_vectors[b],
            ("ngram", int(b)),
        )
        for b in buckets
    ]


def _known(pool, token):
    scale = 1.0 if pool.scales is None else float(pool.scales[token])
    return (float(pool.weights[token]), scale, pool.vectors[token], ("word", int(token)))


def contributions(spec, batch):
    """Per sentence: the factors g (word weight, or (1/m) * oov_weight for an ngram), the
    scales s (1 for float32 and float16) and the stored values y of every contribution,
    in float64, and the indices of their rows in the word and ngram tables."""
    pool = spec.pool_input()
    k = 0
    for s in range(batch.n_sentences):
        found = []
        for token in batch.tokens[batch.offsets[s] : batch.offsets[s + 1]]:
            if token == OOV_TOKEN:
                found += _unknown(
                    pool, batch.ngrams[batch.ngram_offsets[k] : batch.ngram_offsets[k + 1]]
                )
                k += 1
            else:
                found.append(_known(pool, token))
        n = int(batch.offsets[s + 1] - batch.offsets[s])
        g = np.array([f[0] for f in found])
        scale = np.array([f[1] for f in found])
        dim = pool.vectors.shape[1]
        y = np.array([np.asarray(f[2], np.float64) for f in found]).reshape(len(found), dim)
        yield n, g, scale, y, [f[3] for f in found]


def folded(spec, batch, folded_rows, r, a64, b64, a, b, z):
    """The bound of folded logits ``z`` (n, k) against scikit-learn's ``A64 r + b64`` on
    fse's unfolded vectors ``r`` (raw means); ``folded_rows`` maps ("word", i) and
    ("ngram", i) to the folded table rows (k values)."""
    a64, b64 = np.asarray(a64, np.float64), np.asarray(b64, np.float64)
    a, b, z = (np.asarray(x, np.float64) for x in (a, b, z))
    out = np.zeros(z.shape)
    for s, (n, g, scale, y, where) in enumerate(contributions(spec, batch)):
        if n == 0 or not len(g):
            out[s] = U * np.abs(b64) + U * (np.abs(z[s]) + np.abs(b)) + U * np.abs(z[s])
            continue
        c = (g * scale)[:, None] * y  # exact contributions, (K, d)
        r_exact = c.sum(axis=0) / n
        e = gamma(len(g) + 3) * np.abs(c).sum(axis=0) / n
        t = np.array([folded_rows(kind, i) for kind, i in where], np.float64)  # (K, k)
        fold = gamma(chain(a.shape[1]) + 1) * (np.abs(c) @ np.abs(a).T).sum(axis=0) / n
        pool = gamma(len(g) + 2) * (np.abs(g)[:, None] * np.abs(t)).sum(axis=0) / n
        coefficients = U * (np.abs(a64) @ np.abs(r_exact)) + U * np.abs(b64)
        own = fold + pool + coefficients + U * (np.abs(z[s]) + np.abs(b))
        out[s] = own + np.abs(a64) @ e
    return out + _float64_error(a64, np.asarray(r, np.float64))


def allowed_label_changes(z_star, bound):
    """The rows whose label may differ from scikit-learn's within ``bound``: for one output
    where |z*| <= B, else where the gap of the two largest z* is at most the sum of their
    bounds."""
    z_star = np.asarray(z_star, np.float64)
    if z_star.shape[1] == 1:
        return np.abs(z_star[:, 0]) <= bound[:, 0]
    order = np.argsort(-z_star, axis=1, kind="stable")
    rows = np.arange(len(z_star))
    top, second = order[:, 0], order[:, 1]
    gap = z_star[rows, top] - z_star[rows, second]
    return gap <= bound[rows, top] + bound[rows, second]


def proba_tolerance(link, bound, k):
    """The largest difference of probabilities per row: the logits' bound through the
    link (sigmoid moves by at most |dz| / 4, softmax by max |dz| / 2, ovr by 2 max |dsigma|
    / sum sigma, bounded by max |dz| / (2 min-sum)) plus the rounding of the link."""
    worst = bound.max(axis=1)
    if link == "sigmoid@1":
        return worst / 4 + 8 * U
    if link == "softmax@1":
        return worst / 2 + (k + 8) * U
    return None  # ovr: tested against its own float64 formula
