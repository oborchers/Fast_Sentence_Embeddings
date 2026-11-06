"""Every backend stores the bits of the same run without a sink, for each kind of head, each
stored output and each path."""

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support import scores as sc

F32 = np.float32


def bits(x):
    return np.ascontiguousarray(x, dtype=F32).view(np.uint32)


def _group():
    enc = sh.encoder()
    return fse.Heads(
        quality=sh.regressor(enc, vector=True),
        pair=sh.regressor(enc, 2, seed=3),
        flag=sh.classifier(enc, 1, seed=5),
        topic=sh.classifier(enc, 4, seed=6),
        toxic=sh.classifier(enc, 3, link="multilabel", labels=["a", "b", "c"], seed=7),
        plain=sh.classifier(enc, 3, link="none", seed=8),
        normed=sh.classifier(enc, 3, seed=9, link="ovr", normalized=True),
    )


def _store(heads, kind):
    """What each head stores: its decision or values, or its probabilities where it has some."""
    out = {}
    for name, model in heads.items():
        regression = model.TASK == "regression"
        out[name] = "values" if regression else kind
        if kind == "proba" and (regression or model._content.link is None):
            out[name] = "values" if regression else "decision"
    return out


def _columns(info, scores):
    return {h.name: scores[:, h.columns[0] : h.columns[1]] for h in info.heads}


@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("kind", ["decision", "proba"])
@pytest.mark.parametrize("vectors", [False, True])
def test_the_stored_columns_are_the_bits_of_the_run_without_a_sink(
    tmp_path, backend, kind, vectors
):
    heads, texts = _group(), sh.texts(37)
    stored = _store(heads, kind)
    with fse.ScoreSink(tmp_path / "s", heads, store=stored, shard_rows=16) as sink:
        list(
            heads.predict_batches(
                [texts[:20], texts[20:]],
                score_sink=sink,
                return_vectors=vectors,
                backend=backend,
                workers=3,
            )
        )
    info, scores, _ = sc.read_back(tmp_path / "s")
    (want,) = heads.predict_batches(
        [texts], output=stored, return_vectors=vectors, backend=backend, workers=1
    )
    for name, columns in _columns(info, scores).items():
        np.testing.assert_array_equal(
            bits(columns), bits(want.outputs[name]).reshape(len(texts), -1)
        )
    assert {h.path for h in info.heads} <= {"folded", "unfolded", "normalized"}


def test_both_backends_store_the_same_bits(tmp_path):
    heads, texts = _group(), sh.texts(40)
    stored = _store(heads, "decision")
    for backend in ("cpu", "numpy"):
        with fse.ScoreSink(tmp_path / backend, heads, store=stored) as sink:
            list(heads.predict_batches([texts], score_sink=sink, backend=backend))
    np.testing.assert_array_equal(
        bits(sc.read_back(tmp_path / "cpu")[1]), bits(sc.read_back(tmp_path / "numpy")[1])
    )
