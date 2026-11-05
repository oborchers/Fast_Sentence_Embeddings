"""The description of a Model2Vec head both importers share (fse._head._model2vec): which
heads are linear classifiers fse reads, and every refusal that depends on head_config or
the shapes of the tensors."""

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._head import _model2vec as m2v
from support import model2vec_head as mh

F32 = ("float32", (4, 16))


def _spec(activation="softmax", classes=("a", "b", "c", "d")):
    return m2v.spec(1, activation, list(classes), "x", ValueError)


@pytest.mark.parametrize(
    ("activation", "link"), [("softmax", "softmax"), ("sigmoid", "multilabel")]
)
def test_the_activation_gives_the_link(activation, link):
    head = _spec(activation)
    assert (head.link, head.activation, head.labels) == (link, activation, list("abcd"))


@pytest.mark.parametrize("layers", [2, 3])
def test_a_head_with_hidden_layers_is_refused(layers):
    with pytest.raises(UnsupportedArtifactError, match=r"hidden layers with ReLU.*n_layers=0"):
        m2v.spec(layers, "softmax", ["a", "b"], "x", ValueError)


def test_a_head_without_a_layer_is_no_linear_head():
    with pytest.raises(UnsupportedArtifactError, match="no layer"):
        m2v.spec(0, "softmax", ["a", "b"], "x", ValueError)


@pytest.mark.parametrize("layers", [True, 1.0, "1", None, -1])
def test_a_malformed_layer_count(layers):
    with pytest.raises(CorruptArtifactError, match="n_layers must be a non-negative int"):
        m2v.spec(layers, "softmax", ["a"], "x", CorruptArtifactError)


@pytest.mark.parametrize(
    ("activation", "error", "match"),
    [
        ("identity", UnsupportedArtifactError, "regression or projection head"),
        ("relu", UnsupportedArtifactError, "unknown"),
        (3, ValueError, "must be a str"),
        (None, ValueError, "must be a str"),
    ],
)
def test_other_activations(activation, error, match):
    with pytest.raises(error, match=match):
        m2v.spec(1, activation, ["a", "b"], "x", ValueError)


@pytest.mark.parametrize("classes", [None, [], "ab", ("a", "b")])
def test_classes_are_a_non_empty_list(classes):
    with pytest.raises(ValueError, match="classes must be a non-empty list"):
        m2v.spec(1, "softmax", classes, "x", ValueError)


def test_a_layer_of_the_right_description():
    head = _spec()
    assert m2v.check_layer(F32, ("float32", (4,)), head, "x", ValueError) == 4
    assert "Model2Vec StaticModelPipeline (softmax, 4 classes)" in m2v.note(
        head, "StaticModelPipeline", 4
    )


@pytest.mark.parametrize("dtype", ["float64", "float16", "int64", "uint8"])
@pytest.mark.parametrize("which", ["weight", "bias"])
def test_only_float32_is_read(dtype, which):
    weight = (dtype if which == "weight" else "float32", (4, 16))
    bias = (dtype if which == "bias" else "float32", (4,))
    with pytest.raises(UnsupportedArtifactError, match=f"head {which} is {dtype}"):
        m2v.check_layer(weight, bias, _spec(), "x", ValueError)


@pytest.mark.parametrize(
    ("weight", "bias"),
    [
        ((4, 16), ()),  # a scalar bias would be broadcast
        ((4, 16), (4, 1)),
        ((16,), (4,)),
        ((4, 16, 1), (4,)),
        ((4, 16), (3,)),
        ((3, 16), (3,)),  # four classes
        ((0, 16), (0,)),
    ],
)
def test_shapes_that_do_not_describe_one_head(weight, bias):
    with pytest.raises(CorruptArtifactError, match="x:"):
        m2v.check_layer(("float32", weight), ("float32", bias), _spec(), "x", CorruptArtifactError)


def test_softmax_of_one_class_is_constant():
    head = _spec("softmax", ["a"])
    with pytest.raises(UnsupportedArtifactError, match="constant"):
        m2v.check_layer(("float32", (1, 16)), ("float32", (1,)), head, "x", ValueError)
    head = _spec("sigmoid", ["a"])  # one multilabel output is an ordinary head
    assert m2v.check_layer(("float32", (1, 16)), ("float32", (1,)), head, "x", ValueError) == 1


def test_a_winner_change_through_a_third_class():
    """The two largest logits (1, 0.9) are 0.1 apart, beyond the bounds 0.001 each, yet a
    third class of a large bound (2) may win: the check compares the actual winners."""
    z = np.array([[1.0, 0.9, 0.0]])
    bound = np.array([[0.001, 0.001, 2.0]])
    reference, got = np.array([0]), np.array([2])
    assert mh.winner_change_allowed(z, bound, reference, got).all()
    assert not mh.winner_change_allowed(z, bound, reference, np.array([1])).all()
    from support import head_bound

    assert not head_bound.allowed_label_changes(z, bound).all()  # the old helper would refuse
    assert mh.winner_change_allowed(z, bound, reference, reference).all()
