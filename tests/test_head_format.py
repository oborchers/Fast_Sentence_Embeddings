"""The format rules of linear_head 1: the keys and tensors of a head, and the adapter that
hands the encoder part to the encoder's own parser."""

from pathlib import Path

import numpy as np
import pytest

from fse import _reference_head as ref
from fse._container import CHECKSUM_PREFIX, CorruptArtifactError, checksum
from fse._contract import Link, UnsupportedArtifactError
from fse._encoder._format import from_artifact as encoder_from_artifact
from fse._head._content import HeadContent
from fse._head._format import encoder_fingerprint, from_artifact, to_artifact
from support import head as support
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
        _content(k=6, standalone=True),
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
        (_set("fse.head.flat", "true"), ValueError, "exactly for regression heads of one"),
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
    tensors, metadata = _artifact(_content(k=6, standalone=True))
    metadata["fse.head.normalized"] = "true"
    with pytest.raises(ValueError, match="standalone head reads raw means"):
        from_artifact(tensors, metadata)
    normalizing = finished(normalize=True)
    tensors, metadata = _artifact(_content(k=6, standalone=True), normalizing)
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


GOLDEN = Path(__file__).parent / "test_data" / "v2" / "finished"
SENTENCES = ["the é", "日本 thé zz", "", "THE"]
PINNED = {
    "classifier_float32": (
        "xxh3-128:093709742bf5fefb96e336658e9ec3bc",
        "xxh3-128:363c1e166b680245adb9fbb9df929d96",  # encoder_float32.safetensors
        [
            [1057591021, 1040328411, 1051446967],
            [1015122304, 1039477409, 1063079520],
            [1050920551, 1049899156, 1053296902],
            [1057591021, 1040328411, 1051446967],
        ],
    ),
    "regressor_int8": (
        "xxh3-128:beddac876d0c5250988361e64596501e",
        "xxh3-128:900de12daf5acd9cd72ee4368740e1c4",
        [
            [1085684857, 3212886403],
            [3232074522, 3210018643],
            [1065353216, 3212836864],
            [3210084698, 3215638571],
        ],
    ),
    "classifier_standalone": (
        "xxh3-128:117ae14aed7a477e442f16b5527e512f",
        "xxh3-128:6e7b98ccbcf439933966aba688535d62",  # the folded table
        [
            [1044041156, 1062292623],
            [1065209310, 1007454356],
            [1054878310, 1058007757],
            [1044041156, 1062292623],
        ],
    ),
}


@pytest.mark.parametrize("name", PINNED)
def test_golden_heads(name):
    import fse

    fingerprint, encoder, outputs = PINNED[name]
    cls = fse.Regressor if name.startswith("regressor") else fse.Classifier
    model = cls.load(GOLDEN / f"{name}.safetensors")
    assert model.fingerprint == fingerprint
    assert model.encoder_fingerprint == encoder
    got = model.predict(SENTENCES) if cls is fse.Regressor else model.predict_proba(SENTENCES)
    assert got.view(np.uint32).tolist() == outputs
    if not model.standalone:  # the pinned bits are the reference's
        want = support.logits(
            model.encoder._spec,
            model.coef_,
            model.intercept_,
            SENTENCES,
            normalized=model.normalized,
        )
        want = want if cls is fse.Regressor else ref.proba(want, model._content.link)
        assert want.view(np.uint32).tolist() == outputs


def test_the_golden_classifier_reads_the_golden_encoder():
    import fse

    model = fse.Classifier.load(GOLDEN / "classifier_float32.safetensors")
    encoder = fse.Encoder.load(GOLDEN / "encoder_float32.safetensors")
    assert model.encoder.fingerprint == encoder.fingerprint
    assert model.classes_.tolist() == ["a", "b", "c"]
    assert (model.normalized, model.link) == (True, "softmax@1")


def test_a_regression_head_of_one_output_records_its_shape():
    for flat in (True, False):
        content = HeadContent(
            "regression", np.ones((1, 6), F32), np.zeros(1, F32), False, None, None, flat=flat
        )
        tensors, metadata = _artifact(content)
        assert metadata["fse.head.flat"] == str(flat).lower()
        assert from_artifact(tensors, metadata)[0].flat is flat
        del metadata["fse.head.flat"]
        with pytest.raises(ValueError, match="exactly for regression heads of one output"):
            from_artifact(tensors, metadata)


@pytest.mark.parametrize(
    ("encoder", "k", "match"),
    [
        (lambda: finished(normalize=False), 3, "3 outputs has 3 values per row, not '6'"),
        (lambda: finished("int8", normalize=False, dim=3), 3, "is float32"),
        (lambda: finished("float16", normalize=False, dim=2), 2, "is float32"),
    ],
)
def test_a_standalone_file_must_hold_a_folded_table(encoder, k, match):
    """Files with valid checksums whose encoder part is no folded table of the head."""
    tensors, metadata = _artifact(_content(k=k, standalone=True), encoder())
    with pytest.raises(ValueError, match=match):
        from_artifact(tensors, metadata)


def test_a_folded_table_has_no_scales():
    from fse._head._format import _check_folded

    metadata = {"fse.steps": "[]", "fse.output_dim": "2", "fse.dim": "2", "fse.dtype": "float32"}
    with pytest.raises(ValueError, match="has no scales"):
        _check_folded({"scales": np.ones(3, F32)}, metadata, 2)
