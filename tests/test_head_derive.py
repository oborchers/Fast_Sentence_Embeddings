"""The outputs of a head derived from its logits on data alone."""

import numpy as np
import pytest

from fse._backends import get_backend
from fse._contract import Head, Link
from fse._head._derive import decide, derive, proba_width

BACKENDS = ["cpu", "numpy"]


def bits(array):
    return np.ascontiguousarray(array).tobytes()


def test_the_width_of_the_probabilities():
    assert proba_width(1, Link.SIGMOID) == 2
    assert proba_width(1, Link.SOFTMAX) == 2
    assert proba_width(3, Link.SOFTMAX) == 3
    assert proba_width(1, Link.MULTILABEL) == 1
    assert proba_width(4, Link.MULTILABEL) == 4
    with pytest.raises(ValueError, match="no link"):
        proba_width(2, None)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logits_are_given_back_unchanged(backend):
    z = np.array([[0.5, -1.5]], np.float32)
    out = derive(
        get_backend(backend),
        z,
        "logits",
        rule=Head.LINEAR,
        link=Link.SOFTMAX,
        thresholds=None,
        first_row=0,
    )
    assert out is z


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("link", "k"), [(Link.SIGMOID, 1), (Link.SOFTMAX, 1), (Link.SOFTMAX, 3), (Link.OVR, 3)]
)
def test_probabilities_and_linear1_labels_are_those_of_the_backend(backend, link, k):
    host = get_backend(backend)
    z = np.random.default_rng(3).normal(0, 4, (7, k)).astype(np.float32)
    proba = derive(host, z, "proba", rule=Head.LINEAR, link=link, thresholds=None, first_row=0)
    want = np.empty((7, proba_width(k, link)), np.float32)
    host.outputs(z, "proba", link, want, 0)
    assert bits(proba) == bits(want)
    labels = derive(host, z, "labels", rule=Head.LINEAR, link=link, thresholds=None, first_row=0)
    expected = np.empty(7, np.int64)
    host.outputs(z, "labels", None, expected, 0)
    assert bits(labels) == bits(expected)


def test_both_backends_derive_the_same_bits():
    z = np.random.default_rng(5).normal(0, 6, (50, 4)).astype(np.float32)
    kwargs = {"rule": Head.LINEAR2, "link": Link.SOFTMAX, "thresholds": None, "first_row": 0}
    for kind in ("proba", "labels"):
        a = derive(get_backend("cpu"), z, kind, **kwargs)
        b = derive(get_backend("numpy"), z, kind, **kwargs)
        assert bits(a) == bits(b)


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear2_labels_of_a_multilabel_head_are_strictly_above_the_cutoff(backend):
    host = get_backend(backend)
    # Logits of exactly zero and of a tiny positive value give the probability 0.5.
    z = np.array([[0.0, -0.0, 1e-9, 1.0]], np.float32)
    limits = np.full(4, 0.5, np.float32)
    labels = derive(
        host, z, "labels", rule=Head.LINEAR2, link=Link.MULTILABEL, thresholds=limits, first_row=0
    )
    assert labels.dtype == np.bool_
    assert labels.tolist() == [[False, False, False, True]]
    low = np.full(4, 0.25, np.float32)
    again = derive(
        host, z, "labels", rule=Head.LINEAR2, link=Link.MULTILABEL, thresholds=low, first_row=0
    )
    assert again.tolist() == [[True, True, True, True]]


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear2_single_label_decides_on_the_probabilities(backend):
    host = get_backend(backend)
    z = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, 0.0]], np.float32)
    labels = derive(
        host, z, "labels", rule=Head.LINEAR2, link=Link.SOFTMAX, thresholds=None, first_row=0
    )
    assert labels.tolist() == [1, 0]
    assert labels.dtype == np.int64


def test_a_multilabel_head_needs_its_thresholds():
    with pytest.raises(ValueError, match="thresholds"):
        decide(get_backend("cpu"), np.zeros((1, 2), np.float32), Link.MULTILABEL, None)


def test_an_error_names_the_row_of_the_run():
    z = np.array([[0.0], [np.inf]], np.float32)
    with pytest.raises(ValueError, match="row 11"):
        derive(
            get_backend("cpu"),
            z,
            "proba",
            rule=Head.LINEAR,
            link=Link.SIGMOID,
            thresholds=None,
            first_row=10,
        )
