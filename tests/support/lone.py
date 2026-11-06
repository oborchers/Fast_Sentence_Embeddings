"""Single heads of every kind, and the store directories two runs of them leave."""

from pathlib import Path

import numpy as np

import fse
from support import score_heads as sh

F32 = np.float32
NAME = "m"


def enc():
    return sh.encoder()


def kinds(encoder=None):
    """(id, model, what it can store): one head of each kind."""
    e = encoder or enc()
    multi = sh.classifier(
        e, 3, link="multilabel", labels=["a", "b", "c"], thresholds=[0.3, 0.5, 0.7]
    )
    rng = np.random.default_rng(5)
    calibrated = fse.Classifier.from_arrays(
        e,
        rng.standard_normal((1, e.output_dim)).astype(F32),
        rng.standard_normal(1).astype(F32),
        labels=["no", "yes"],
        link="softmax",
        temperature=2.0,
    )
    return [
        ("softmax", sh.classifier(e, 3, seed=4), ("decision", "proba")),
        ("normalized", sh.classifier(e, 3, seed=4, normalized=True), ("decision", "proba")),
        ("multilabel", multi, ("decision", "proba")),
        ("linear2", calibrated, ("decision", "proba")),
        ("none", sh.classifier(e, 3, link="none"), ("decision",)),
        ("flat", sh.regressor(e, 1, vector=True), ("values",)),
        ("targets", sh.regressor(e, 3), ("values",)),
    ]


def standalone(model, tmp_path):
    """The model saved standalone and loaded again."""
    model.save(tmp_path / "standalone.safetensors", standalone=True)
    return type(model).load(tmp_path / "standalone.safetensors")


def batches(n=40, size=13, ids=False):
    texts = sh.texts(n)
    chunks = [texts[i : i + size] for i in range(0, n, size)]
    if not ids:
        return chunks
    return [(c, np.arange(i * size, i * size + len(c)) * 3 + 1) for i, c in enumerate(chunks)]


def files(path):
    """Every file of a store directory: relative path to bytes."""
    root = Path(path)
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def same_store(one, other):
    """The directories hold the same files with the same bytes."""
    a, b = files(one), files(other)
    assert a.keys() == b.keys()
    for name in a:
        assert a[name] == b[name], name


def outputs_bits(batch_list):
    """The outputs of batches as comparable bytes."""
    return [None if b.outputs is None else np.asarray(b.outputs).tobytes() for b in batch_list]
