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

from itertools import pairwise

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

    if job["kind"] == "heads-kind":
        head = kind_head(job)
        head.predict_proba(["warm the folded table"], workers=job["workers"])
        return lambda texts: head.predict_proba(texts, workers=job["workers"])
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


def kind_head(job):
    """Four independent labels or a temperature-calibrated four-class head, without fitting."""
    from fse import Classifier

    encoder = _encoder(job)
    weight, bias = _arrays(0, encoder.output_dim)
    extra = {"temperature": 2.0} if job["head_kind"] == "calibrated" else {}
    return Classifier.from_arrays(
        encoder,
        weight,
        bias,
        labels=[f"class_{i}" for i in range(SPEC["outputs"])],
        link="multilabel" if job["head_kind"] == "multilabel" else "softmax",
        normalized=False,
        **extra,
    )


def check_kind(job, texts, sample, indices):
    """Every measured arm checks its fixed 10,000 rows against fse._reference, untimed."""
    if job["kind"] != "heads-kind":
        return {}
    from _speed_cells import check_output

    expected_rows = min(10_000, job["texts"])
    if len(indices) != expected_rows:
        raise ValueError("head-kind sample must contain 10,000 rows (all rows in small probes)")
    reference = kind_head(job).predict_proba([texts[i] for i in indices], backend="numpy")
    check_output(reference, sample, exact=True)
    return {"sample_numpy_bitwise_equal": True, "sample_checked_rows": len(indices)}


def sample_indices(job):
    """Check new head sample positions in their original JSON form before conversion."""
    from pathlib import Path

    import numpy as np
    from _harness.records import document

    raw = document(Path(job["sample_file"]))
    if job["kind"] in ("heads-kind", "heads-sinks"):
        if not isinstance(raw, list) or len(raw) != min(10_000, job["texts"]):
            raise ValueError(
                "head sample needs 10,000 ordered positions (all rows in small probes)"
            )
        if any(type(i) is not int or not 0 <= i < job["texts"] for i in raw):
            raise ValueError("head sample positions must be integers in the input range")
        if any(a >= b for a, b in pairwise(raw)):
            raise ValueError("head sample positions must be unique and increasing")
    return np.asarray(raw, dtype=np.int64)
