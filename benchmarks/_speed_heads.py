"""Linear heads in the publication study: three fixed pairs with one output contract.

The workload is the fixed head of the development study (``linear-heads@1``: four softmax
outputs, float32 PCG64 seed 17, ``fse.Classifier.from_arrays``, no training). Every arm returns
the probabilities (output ``proba``) of the heads in the order of their names, for rows that
are the raw means of the table (the encoder never normalizes), as one float32 matrix of
``dimensions`` columns: 4 for one head, 16 for the four heads ``head_0`` to ``head_3`` side by
side. Folding the head into the table happens before the first timed call: the factory makes
every lazy table and plan, and the child calls the arm once untimed before its timer.

Pairs: ``fse`` (folded ``predict_proba``) against ``model2vec`` (``encode`` with
``normalize=False``, then the affine map and the softmax in NumPy); ``fse`` against
``fse-unfolded`` (``predict_batches`` with the vectors); four heads in one ``Heads.predict``
(``fse``) against four single ``predict_proba`` calls (``fse-separate``).
"""

from __future__ import annotations

from _development_heads import SPEC
from _development_heads_multi import models as head_group

OUTPUT = "proba"


def _arrays(index, width):
    import numpy as np

    rng = np.random.default_rng(SPEC["seed"] + index)
    weight = rng.standard_normal((SPEC["outputs"], width)).astype(np.float32)
    return weight, rng.standard_normal(SPEC["outputs"]).astype(np.float32)


def _check(job):
    if job["kind"] != "heads" or job["dimensions"] != SPEC["outputs"] * job["heads"]:
        raise ValueError("a head cell has kind heads and dimensions = outputs * heads")
    if job["heads"] not in (1, 4):
        raise ValueError("the head cells fix one head or four")


def _encoder(job):
    from fse import Encoder

    return Encoder.from_pretrained(job["artifact"], normalize=False, mmap=True)


def _group(job):
    return head_group(_encoder(job), {**job, "stage": "head-group-folded"})


def fse_head(job):
    """The arms on fse: ``fse`` (folded, or one pass), ``fse-unfolded`` and ``fse-separate``."""
    import numpy as np

    from fse import Heads

    _check(job)
    group = _group(job)
    options = {"workers": job["workers"], "backend": "cpu"}
    label = job["label"]
    if job["heads"] == 1:
        (head,) = group.values()
        if label == "fse-unfolded":

            def predict(texts):
                (batch,) = head.predict_batches(
                    (texts,), output=OUTPUT, return_vectors=True, **options
                )
                return batch.outputs

        else:

            def predict(texts):
                return head.predict_proba(texts, **options)

    elif label == "fse-separate":

        def predict(texts):
            return np.concatenate(
                [head.predict_proba(texts, **options) for head in group.values()], axis=1
            )

    else:
        heads = Heads(**group)

        def predict(texts):
            got = heads.predict(texts, output=OUTPUT, **options)
            return np.concatenate([got[name] for name in group], axis=1)

    # Every lazy table and plan is made here, outside every timer.
    predict(["warm the folded tables and plans"])
    return predict


def model2vec_head(job):
    """The Model2Vec arm: raw means, then the affine map and the softmax in NumPy."""
    import numpy as np
    from model2vec import StaticModel

    _check(job)
    if job["heads"] != 1:
        raise ValueError("the Model2Vec head arm has one head")
    model = StaticModel.from_pretrained(job["snapshot"], normalize=False)
    weight, bias = _arrays(0, job["vector_dimensions"])

    def predict(texts):
        means = model.encode(texts, use_multiprocessing=job["workers"] > 1)
        logits = means.astype(np.float32, copy=False) @ weight.T + bias
        exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)

    return predict
