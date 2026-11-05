"""A group of heads for the tests of score stores: a regressor, a multilabel head and a
single-label head on one encoder."""

import numpy as np

import fse
from support.encoder import finished

F32 = np.float32


def encoder(**kwargs):
    return finished(
        "float16", fasttext=kwargs.pop("fasttext", True), **{"normalize": False, **kwargs}
    )


def classifier(enc, k=3, *, link="auto", seed=None, labels=None, thresholds=None, normalized=False):
    rng = np.random.default_rng(k if seed is None else seed)
    weight = rng.standard_normal((k, enc.output_dim)).astype(F32)
    names = labels or (["no", "yes"] if k == 1 else [f"c{i}" for i in range(k)])
    return fse.Classifier.from_arrays(
        enc,
        weight,
        rng.standard_normal(k).astype(F32),
        labels=names,
        link=link,
        thresholds=thresholds,
        normalized=normalized,
    )


def regressor(enc, targets=1, *, seed=9, vector=False):
    """A regressor; ``vector=True`` gives one target as a weight vector (it is then flat)."""
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((targets, enc.output_dim)).astype(F32)
    return fse.Regressor.from_arrays(
        enc, weight[0] if vector else weight, rng.standard_normal(targets).astype(F32)
    )


def group(enc=None, **kwargs):
    """quality (regressor), toxic (multilabel, three labels) and topic (softmax, three)."""
    enc = enc or encoder(**kwargs)
    toxic = classifier(
        enc, 3, link="multilabel", labels=["a", "b", "c"], thresholds=[0.3, 0.5, 0.7]
    )
    return fse.Heads(quality=regressor(enc), toxic=toxic, topic=classifier(enc, 3, seed=4))


def texts(n=40):
    return [" ".join(f"w{(i * 7 + j) % 150}" for j in range(3 + i % 5)) for i in range(n)]
