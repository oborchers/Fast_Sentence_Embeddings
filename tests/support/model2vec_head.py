"""Model2Vec classifier pipelines written by model2vec 0.10.0 itself, and the bound of fse's
head against theirs.

A fixture is a ``StaticModelPipeline`` built from a tiny ``StaticModel`` (the vocabularies of
tests/support) and an ``MLPHead`` of one linear layer, saved with ``save_pretrained``.
Nothing is downloaded and no weights are checked in.

The bound. Let r be fse's vector of a text and r' model2vec's, z' = A r' + b the exact
float64 logits of r', and u = 2**-24, gamma(m) = m u / (1 - m u).

- fse against its exact logits of r: ``head_bound.unfolded`` (normalized heads) or
  ``head_bound.folded`` (raw means), which also cover the pooling of a folded table;
- r against r': ``converted.row_bounds`` in the 2-norm, so ``|a_j (r - r')| <= |a_j|_2``
  times that bound;
- model2vec's own product ``x @ W.T + b`` in float32, in any order of summation:
  ``gamma(d + 1) sum|a||r'| + u (|z| + |b|)``.

The three add to the bound B of ``|z_fse - z_m2v|`` for every row and output. A label may
differ from model2vec's only where this bound allows a swap of the winners
(``winner_change_allowed``).
"""

import numpy as np

from fse import _reference
from fse import _reference_head as ref
from fse._contract import Batch, Normalize
from support import converted as cv
from support import head as head_support
from support import head_bound, unigram
from support import model2vec as mv
from support import wordpiece as wp

U = 2.0**-24
F32 = np.float32


def texts(tokenizer="wordpiece", count=40):
    if tokenizer == "wordpiece":
        return [*wp.corpus(wp.vocabulary(), count=count), "", "zzzzzz", wp.vocabulary()[7]]
    return [*unigram.corpus(count), ""]


def static_model(*, normalize=True, tokenizer="wordpiece", max_length=512, seed=0):
    """A model2vec StaticModel with a table of random float32 rows."""
    from model2vec import StaticModel

    if tokenizer == "wordpiece":
        tokens = wp.vocabulary()
        hf = wp.hf_tokenizer(tokens, {t: tokens.index(t) for t in wp.SPECIAL if t in tokens})
        table = mv.rows(len(tokens), seed=seed)
    else:
        vocab = unigram.pieces_and_scores()
        hf = unigram.hf_tokenizer(vocab)
        table = mv.rows(len(vocab), seed=seed + 1)
    return StaticModel(table, hf, normalize=normalize, max_length=max_length)


def layer(k, *, seed=0, scale=1.0, dtype=F32):
    rng = np.random.default_rng(100 + seed)
    weight = (scale * rng.standard_normal((k, mv.DIM))).astype(dtype)
    return weight, rng.standard_normal(k).astype(dtype)


def pipeline(
    activation="softmax",
    k=3,
    *,
    normalize=True,
    tokenizer="wordpiece",
    classes=None,
    seed=0,
    scale=1.0,
    layers=None,
    model=None,
):
    """A StaticModelPipeline of one linear layer (or of ``layers``, a list of (weight,
    bias) pairs) with the activation ``softmax``, ``sigmoid`` or ``identity``."""
    from model2vec.inference import StaticModelPipeline
    from model2vec.inference.mlp import Activation, Layer, MLPHead

    layers = [layer(k, seed=seed, scale=scale)] if layers is None else layers
    names = np.array(["a", "b", "c", "d"][:k] if classes is None else classes)
    model = static_model(normalize=normalize, tokenizer=tokenizer) if model is None else model
    head = MLPHead(
        [Layer(w, b) for w, b in layers],
        Activation(activation),
        None if activation == "identity" else names,
    )
    return StaticModelPipeline(model, head)


def saved(folder, **kwargs):
    """The folder of ``pipeline(**kwargs).save_pretrained``, and the pipeline."""
    pipe = pipeline(**kwargs)
    pipe.save_pretrained(str(folder))
    return folder, pipe


def _batch(encoder, sentences):
    """The token ids fse pools for ``sentences``, as a Batch."""
    ids = cv.ids_of(encoder, sentences)
    offsets = np.concatenate([[0], np.cumsum([len(i) for i in ids])]).astype(np.int64)
    tokens = np.concatenate([np.asarray(i, np.uint32) for i in ids] or [np.empty(0, np.uint32)])
    return Batch(0, offsets, tokens)


def reference_logits(clf, sentences):
    """The logits of fse's reference (fse._reference and fse._reference_head) for the head
    of ``clf`` on the ids fse pools: the bits every backend must give."""
    pool = clf.encoder._spec.pool_input()
    a, b = clf.coef_, clf.intercept_
    batch = _batch(clf.encoder, sentences)
    if not clf.normalized:
        rows = _reference.pool_weighted_mean(batch, *head_support.folded_pool(pool, a))
        return ref.logits(rows, None, b, 0)
    args = (pool.vectors, pool.weights, pool.scales, pool.ngram_vectors, pool.ngram_scales)
    rows = _reference.pool_weighted_mean(batch, *args, pool.oov_weight)
    return ref.logits(_reference.postprocess(rows, (Normalize(),), 0), a, b, 0)


def exact(clf, pipe, sentences):
    """The logits of fse and of model2vec, the exact float64 logits of model2vec's vectors
    and the bound B of fse against model2vec (n, k)."""
    a, b = clf.coef_, clf.intercept_
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    r = clf.encoder.encode(sentences)
    r2 = pipe.model.encode(sentences)
    z = clf.decision_function(sentences).reshape(len(sentences), -1).astype(np.float64)
    z2 = pipe.head._logits(r2).astype(np.float64)
    z_star = r2.astype(np.float64) @ a64.T + b64
    spec = clf.encoder._spec
    if clf.normalized:
        fse_bound = head_bound.unfolded(r, a64, b64, a, b, z)
    else:
        table = head_support.folded_pool(spec.pool_input(), a)
        batch = _batch(clf.encoder, sentences)

        def rows(kind, i):
            return table[0][i] if kind == "word" else table[3][i]

        fse_bound = head_bound.folded(spec, batch, rows, r, a64, b64, a, b, z)
    row = np.array(
        [0.0 if x is None else x for x in cv.row_bounds(clf.encoder, sentences, clf.normalized)]
    )
    own = np.linalg.norm(a64, axis=1)[None, :] * row[:, None]
    gamma = (mv.DIM + 1) * U / (1 - (mv.DIM + 1) * U)
    theirs = gamma * (np.abs(r2.astype(np.float64)) @ np.abs(a64).T) + U * (
        np.abs(z2) + np.abs(b64)
    )
    return z, z2, z_star, fse_bound + own + theirs


def winner_change_allowed(z_star, bound, reference, got):
    """Per row: whether fse's winner ``got`` may differ from the reference winner. Both
    winners must lie within the bounds of each other: ``|z_i - z_j| <= B_i + B_j`` for the
    exact logits ``z_star``, which holds also when a third class wins."""
    rows = np.arange(len(z_star))
    gap = np.abs(z_star[rows, reference] - z_star[rows, got])
    return (reference == got) | (gap <= bound[rows, reference] + bound[rows, got])


def proba_tolerance(link, bound):
    """The largest difference of probabilities per row for the logit bound: sigmoid moves
    by at most |dz| / 4, softmax by max |dz| / 2, plus the rounding of the link."""
    worst = bound.max(axis=1)
    return worst / 4 + 16 * U if link == "sigmoid" else worst / 2 + 24 * U


def assert_matches(clf, pipe, sentences, *, threshold=0.5):
    """fse's logits, probabilities and labels against model2vec's own pipeline: logits
    within the bound B, probabilities within the tolerance it implies and within the fixed
    rtol=3e-5 and atol=3e-6 (a gate of the fixtures, not a guarantee for every head), and
    labels equal. Only the calls of model2vec run without numpy's warnings for saturated
    logits (its sigmoid computes both branches)."""
    z, z2, z_star, bound = exact(clf, pipe, sentences)
    assert (np.abs(z - z2) <= bound).all()
    with np.errstate(over="ignore", invalid="ignore"):
        want = pipe.predict_proba(sentences)
        theirs = pipe.predict(sentences)
    got = clf.predict_proba(sentences)
    multilabel = clf.link == "multilabel@1"
    link = "sigmoid" if multilabel else "softmax"
    tolerance = proba_tolerance(link, bound)
    assert (np.abs(got - want).max(axis=1) <= tolerance).all()
    np.testing.assert_allclose(got, want, rtol=3e-5, atol=3e-6)
    labels = clf.predict(sentences)
    if multilabel:
        mask = want > threshold
        assert (labels == mask).all()
        assert [list(theirs[i]) for i in range(len(theirs))] == [
            list(clf.classes_[row]) for row in labels
        ]
    else:
        reference, ours = z2.argmax(axis=1), z.argmax(axis=1)
        assert winner_change_allowed(z_star, bound, reference, ours).all()
        assert (labels == theirs).all()
    return z, bound
