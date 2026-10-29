import numpy as np
import pytest

from fse._contract import (
    EXTERNAL_TOKENIZER,
    OOV_TOKEN,
    TRANSFORM_VERSION,
    Aggregation,
    Batch,
    Normalize,
    Op,
    Tokenizer,
    UnsupportedArtifactError,
    check_steps,
    check_tokenizer,
    check_transform_version,
    parse_id,
    prerelease_error,
)


def test_identifiers_round_trip():
    for member in [*Aggregation, *Op]:
        assert type(member).parse(str(member)) is member
    assert str(Aggregation.WEIGHTED_MEAN) == "weighted_mean@1"
    assert Op.NORMALIZE.id == "normalize@1"
    assert parse_id("max@12") == ("max", 12)


@pytest.mark.parametrize(
    ("cls", "text"),
    [(Aggregation, "max@1"), (Aggregation, "weighted_mean@2"), (Op, "normalize@9")],
)
def test_unknown_identifiers_are_rejected(cls, text):
    with pytest.raises(UnsupportedArtifactError, match=r"supported: .*@1"):
        cls.parse(text)


@pytest.mark.parametrize(
    "text", ["weighted_mean", "x@0", "x@01", "x@1.0", " x@1", "X@1", "1x@1", "x@", 3, None]
)
def test_malformed_identifiers_raise(text):
    with pytest.raises(ValueError, match="identifier"):
        parse_id(text)
    with pytest.raises(ValueError, match="identifier"):
        Aggregation.parse(text)


def test_transform_version():
    assert check_transform_version(TRANSFORM_VERSION) == 2
    for version in (1, 3):
        with pytest.raises(UnsupportedArtifactError, match="supported: 2"):
            check_transform_version(version)
    for version in ("2", 2.0, True):
        with pytest.raises(ValueError, match="must be an int"):
            check_transform_version(version)


def test_output_dim():
    assert Aggregation.WEIGHTED_MEAN.output_dim(300) == 300
    assert Aggregation.WEIGHTED_MEAN.output_dim(np.int64(3)) == 3
    with pytest.raises(ValueError, match="at least 1"):
        Aggregation.WEIGHTED_MEAN.output_dim(0)
    for value in (True, 1.5, float("nan"), "3"):
        with pytest.raises(TypeError, match="must be an int"):
            Aggregation.WEIGHTED_MEAN.output_dim(value)


def test_step_lists():
    n = Normalize()
    for steps in ([], [n], (n,)):
        assert check_steps(steps, 4) == tuple(steps)
    with pytest.raises(ValueError, match="none or normalize once"):
        check_steps([n, n], 4)
    with pytest.raises(TypeError, match="not a postprocessing step"):
        check_steps(["normalize"], 4)
    with pytest.raises(ValueError, match="at least 1"):
        check_steps([], 0)
    assert n.op is Op.NORMALIZE
    assert list(Op) == [Op.NORMALIZE]


def test_the_error_of_a_pre_release():
    error = prerelease_error("encoder", ["weights", "fse.recipe"])
    assert isinstance(error, UnsupportedArtifactError)
    assert str(error) == (
        "this encoder was written by a pre-release of 2.0 (it holds fse.recipe, weights, "
        "which 2.0 no longer has); rebuild the model with this fse"
    )


def _u32(*values):
    return np.array(values, dtype=np.uint32)


def _i64(*values):
    return np.array(values, dtype=np.int64)


def test_batch_accepts_valid_input():
    batch = Batch(3, _i64(0, 2, 2, 3), _u32(1, 2, 0))
    assert batch.n_sentences == 3
    empty = Batch(0, _i64(0), _u32())
    assert empty.n_sentences == 0
    ft = Batch(0, _i64(0, 2), _u32(OOV_TOKEN, 1), _i64(0, 3), _u32(5, 6, 7))
    assert ft.n_sentences == 1
    Batch(np.int64(2), _i64(0, 1), _u32(OOV_TOKEN), _i64(0, 0), _u32())


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"first_row": -1}, ValueError, "between 0 and"),
        ({"first_row": True}, TypeError, "first_row must be an int"),
        ({"first_row": 1.0}, TypeError, "first_row must be an int"),
        ({"offsets": [0, 1]}, TypeError, "numpy array"),
        ({"offsets": np.array([0, 1], dtype=np.int32)}, TypeError, "int64"),
        ({"offsets": _i64(1, 1)}, ValueError, "start with 0"),
        ({"offsets": _i64()}, ValueError, "start with 0"),
        ({"offsets": _i64(0, 1, 0, 1)}, ValueError, "not decrease"),
        # np.diff of these offsets wraps around to positive values only.
        ({"offsets": _i64(0, 2**63 - 1, -2, 1)}, ValueError, "not decrease"),
        ({"first_row": 2**63}, ValueError, "between 0 and"),
        ({"first_row": 2**63, "offsets": _i64(0), "tokens": _u32()}, ValueError, "between 0 and"),
        ({"first_row": 2**63 - 1, "offsets": _i64(0, 1, 1)}, ValueError, "int64 range"),
        ({"offsets": _i64(0, 2)}, ValueError, "end with 1"),
        ({"offsets": _i64(0, 1).reshape(1, 2)}, ValueError, "one dimension"),
        ({"tokens": np.array([1], dtype=np.int64)}, TypeError, "uint32"),
        ({"tokens": _u32(1, 2, 3, 4)[::2]}, ValueError, "C-contiguous"),
        ({"tokens": _u32(OOV_TOKEN)}, ValueError, "without ngrams"),
        ({"ngram_offsets": _i64(0)}, ValueError, "given together"),
    ],
)
def test_batch_rejects_invalid_input(kwargs, error, match):
    args = {"first_row": 0, "offsets": _i64(0, 1), "tokens": _u32(1), **kwargs}
    with pytest.raises(error, match=match):
        Batch(**args)


def test_batch_checks_ngrams():
    tokens = _u32(OOV_TOKEN, 1, OOV_TOKEN)
    offsets = _i64(0, 3)
    with pytest.raises(ValueError, match="3 entries for 2 unknown words"):
        Batch(0, offsets, tokens, _i64(0, 1), _u32(4))
    with pytest.raises(ValueError, match="end with 2"):
        Batch(0, offsets, tokens, _i64(0, 1, 3), _u32(4, 5))
    with pytest.raises(TypeError, match="uint32"):
        Batch(0, offsets, tokens, _i64(0, 1, 2), np.array([4, 5], dtype=np.int64))


def test_tokenizer_identifiers():
    assert Tokenizer.FSE_WORDS.id == "fse_words@1"
    assert check_tokenizer("fse_words@1") == "fse_words@1"
    assert check_tokenizer("external") == EXTERNAL_TOKENIZER == "external"


@pytest.mark.parametrize(
    "text",
    ["fse_words@2", "nltk@1", "External", "external@1", "", 1, None, np.array(["external", "x"])],
)
def test_unknown_tokenizer_is_rejected(text):
    with pytest.raises(UnsupportedArtifactError, match="supported: external, fse_words@1"):
        check_tokenizer(text)
