"""The format rules of linear_head 1: the keys and tensors of a head, and the adapter that
hands the encoder part to the encoder's own parser."""

import numpy as np
import pytest

from fse._container import CHECKSUM_PREFIX, CorruptArtifactError, checksum
from fse._contract import Link, UnsupportedArtifactError
from fse._encoder._format import from_artifact as encoder_from_artifact
from fse._head._content import HeadContent
from fse._head._format import encoder_fingerprint, from_artifact, to_artifact
from support.encoder import finished

F32 = np.float32


def _content(task="classification", k=3, dim=6, standalone=False, normalized=False):
    rng = np.random.default_rng(0)
    weight = None if standalone else rng.standard_normal((k, dim)).astype(F32)
    bias = rng.standard_normal(k).astype(F32)
    if task == "regression":
        return HeadContent(task, weight, bias, normalized, None, None, standalone)
    names = ("neg", "pos") if k == 1 else tuple(f"c{i}" for i in range(k))
    link = Link.SIGMOID if k == 1 else Link.SOFTMAX
    return HeadContent(task, weight, bias, normalized, link, names, standalone)


def _artifact(content, encoder=None):
    encoder = encoder or finished(normalize=False)
    tensors, metadata, _ = encoder._artifact()
    out, keys = to_artifact(content, tensors, metadata, encoder.fingerprint)
    keys.update({CHECKSUM_PREFIX + k: checksum(v) for k, v in out.items()})  # as a container
    return out, keys


@pytest.mark.parametrize(
    "content",
    [
        _content(),
        _content(k=1),
        _content("regression", k=2, normalized=True),
        _content(standalone=True),
    ],
    ids=["classifier", "binary", "regressor", "standalone"],
)
def test_round_trip(content):
    encoder = finished(normalize=False)
    tensors, metadata = _artifact(content, encoder)
    assert metadata["fse.kind"] == "linear_head"
    assert metadata["fse.head"] == "linear@1"
    assert ("head.weight" in tensors) != content.standalone
    got, enc_tensors, enc_metadata, found = from_artifact(tensors, metadata)
    assert found == encoder.fingerprint == metadata["fse.encoder_fingerprint"]
    assert (got.task, got.normalized, got.link, got.labels, got.standalone) == (
        content.task,
        content.normalized,
        content.link,
        content.labels,
        content.standalone,
    )
    np.testing.assert_array_equal(got.bias.view(np.uint32), content.bias.view(np.uint32))
    if content.weight is not None:
        np.testing.assert_array_equal(got.weight, content.weight)
    assert enc_metadata["fse.kind"] == "encoder"
    assert "head.bias" not in enc_tensors
    assert not any(k.startswith("fse.head") or k.endswith("head.bias") for k in enc_metadata)
    encoder_from_artifact(enc_tensors, enc_metadata)  # the encoder rules apply unchanged


def test_labels_are_canonical_json():
    _, metadata = _artifact(_content())
    assert metadata["fse.head.labels"] == '["c0","c1","c2"]'
    numbers = HeadContent(
        "classification", None, np.zeros(1, F32), False, Link.SIGMOID, (0, 7), True
    )
    assert _artifact(numbers)[1]["fse.head.labels"] == "[0,7]"
    assert "fse.head.labels" not in _artifact(_content("regression"))[1]


def _changed(edit):
    tensors, metadata = _artifact(_content())
    edit(tensors, metadata)
    return tensors, metadata


def _set(key, value):
    def edit(_tensors, metadata):
        if value is None:
            del metadata[key]
        else:
            metadata[key] = value

    return edit


@pytest.mark.parametrize(
    ("edit", "error", "match"),
    [
        (_set("fse.kind", "encoder"), UnsupportedArtifactError, "'encoder', not 'linear_head'"),
        (_set("fse.format_version", "2"), UnsupportedArtifactError, "format version '2'"),
        (_set("fse.head.extra", "1"), UnsupportedArtifactError, "unknown metadata keys"),
        (_set("fse.head", "linear@2"), UnsupportedArtifactError, "head linear@2 is not supported"),
        (_set("fse.head.task", None), ValueError, "needs the metadata key fse.head.task"),
        (_set("fse.head.task", "ranking"), ValueError, "fse.head.task must be one of"),
        (_set("fse.head.normalized", "yes"), ValueError, "fse.head.normalized must be one of"),
        (_set("fse.head.link", "tanh@1"), UnsupportedArtifactError, "link tanh@1"),
        (_set("fse.head.link", "sigmoid@1"), ValueError, "sigmoid takes one output"),
        (_set("fse.head.labels", None), ValueError, "exactly for classification"),
        (_set("fse.head.labels", '["c0", "c1", "c2"]'), ValueError, "canonical JSON list"),
        (_set("fse.head.labels", "[0.5,1,2]"), ValueError, "int or str labels"),
        (_set("fse.head.labels", '["a","a","b"]'), ValueError, "unique"),
        (_set("fse.encoder_fingerprint", "x"), ValueError, "must be a fingerprint"),
        (_set("fse.tokenizer", "external"), CorruptArtifactError, "encoder part of the head"),
        (_set("fse.head.standalone", "true"), ValueError, "holds no head.weight"),
    ],
)
def test_refusals(edit, error, match):
    tensors, metadata = _changed(edit)
    with pytest.raises(error, match=match):
        from_artifact(tensors, metadata)


def _tensor(name, value):
    def edit(tensors, _metadata):
        if value is None:
            del tensors[name]
        else:
            tensors[name] = value

    return edit


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_tensor("head.bias", None), "needs the tensor head.bias"),
        (_tensor("head.weight", None), "needs the tensor head.weight"),
        (_tensor("head.bias", np.zeros(3)), "head.bias must be a non-empty F32"),
        (_tensor("head.weight", np.zeros(6, F32)), "head.weight must be a non-empty F32"),
        (_tensor("head.bias", np.array([0, np.nan, 0], F32)), "not finite"),
        (_tensor("head.bias", np.zeros(2, F32)), "3 rows, but head.bias 2"),
        (_tensor("head.weight", np.zeros((3, 5), F32)), "rows of 5 values, but the encoder"),
    ],
)
def test_tensor_refusals(edit, match):
    tensors, metadata = _changed(edit)
    with pytest.raises(ValueError, match=match):
        from_artifact(tensors, metadata)


def test_a_standalone_head_reads_raw_means_without_steps():
    tensors, metadata = _artifact(_content(standalone=True))
    metadata["fse.head.normalized"] = "true"
    with pytest.raises(ValueError, match="standalone head reads raw means"):
        from_artifact(tensors, metadata)
    normalizing = finished(normalize=True)
    tensors, metadata = _artifact(_content(standalone=True), normalizing)
    with pytest.raises(ValueError, match="folded table of a standalone head has no steps"):
        from_artifact(tensors, metadata)


def test_a_regression_head_has_no_link_and_no_labels():
    tensors, metadata = _artifact(_content("regression"))
    metadata["fse.head.link"] = "softmax@1"
    with pytest.raises(ValueError, match="regression head has no link"):
        from_artifact(tensors, metadata)
    tensors, metadata = _artifact(_content("regression"))
    metadata["fse.head.labels"] = '["a","b","c"]'
    with pytest.raises(ValueError, match="exactly for classification"):
        from_artifact(tensors, metadata)


def test_encoder_fingerprint_is_the_encoders():
    encoder = finished()
    tensors, metadata, _ = encoder._artifact()
    sums = {k: checksum(v) for k, v in tensors.items()}
    assert encoder_fingerprint(tensors, metadata, sums) == encoder.fingerprint
