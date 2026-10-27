import numpy as np
import pytest

from fse._contract import (
    OOV_TOKEN,
    TRANSFORM_VERSION,
    Aggregation,
    Batch,
    Normalize,
    Op,
    Project,
    RemoveComponents,
    UnsupportedArtifactError,
    check_steps,
    check_transform_version,
    output_dim,
    parse_id,
)


def test_identifiers_round_trip():
    for member in [*Aggregation, *Op]:
        assert type(member).parse(str(member)) is member
    assert str(Aggregation.WEIGHTED_MEAN) == "weighted_mean@1"
    assert Op.REMOVE_COMPONENTS.id == "remove_components@1"
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
    with pytest.raises(ValueError, match="at least 1"):
        Aggregation.WEIGHTED_MEAN.output_dim(0)


def _components(k=2, d=4):
    return np.eye(k, d, dtype=np.float32)


def test_step_lists():
    rc, p, n = RemoveComponents(_components()), Project(_components(3, 4)), Normalize()
    for steps in ([], [rc], [p], [n], [rc, p], [rc, n], [p, n], [rc, p, n]):
        assert check_steps(steps, 4) == tuple(steps)
    for steps in ([n, rc], [p, rc], [n, p], [rc, rc], [n, n], [rc, n, n]):
        with pytest.raises(ValueError, match="each at most once"):
            check_steps(steps, 4)
    with pytest.raises(TypeError, match="not a postprocessing step"):
        check_steps(["normalize"], 4)
    with pytest.raises(ValueError, match="remove_components@1 step takes rows of 4 values, not 5"):
        check_steps([rc], 5)
    with pytest.raises(ValueError, match="project@1 step takes rows of 4 values, not 5"):
        check_steps([p], 5)
    # A projection changes the dimension that later steps see.
    assert output_dim([rc, p, n], 4) == 3
    assert output_dim([rc, n], 4) == 4
    assert rc.op is Op.REMOVE_COMPONENTS
    assert p.op is Op.PROJECT
    assert n.op is Op.NORMALIZE


def test_project_validates():
    assert Project(_components(2, 5)).output_dim == 2
    with pytest.raises(ValueError, match="between 1 and 2 rows, not 3"):
        Project(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        Project(np.eye(2))


def test_remove_components_validates():
    RemoveComponents(_components(), np.ones(2, dtype=np.float32))
    with pytest.raises(TypeError, match="numpy array"):
        RemoveComponents([[1.0, 0.0]])
    with pytest.raises(TypeError, match="float32"):
        RemoveComponents(np.eye(2))
    with pytest.raises(ValueError, match="2 dimensions"):
        RemoveComponents(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        RemoveComponents(np.asfortranarray(np.ones((2, 3), dtype=np.float32)))
    with pytest.raises(ValueError, match="NaN"):
        RemoveComponents(np.full((1, 2), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="between 1 and 2 rows, not 3"):
        RemoveComponents(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="between 1 and 2 rows, not 0"):
        RemoveComponents(np.ones((0, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        RemoveComponents(_components(), np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="NaN"):
        RemoveComponents(_components(), np.array([1, np.inf], dtype=np.float32))


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
        ({"first_row": -1}, ValueError, "negative"),
        ({"first_row": True}, TypeError, "first_row must be an int"),
        ({"first_row": 1.0}, TypeError, "first_row must be an int"),
        ({"offsets": [0, 1]}, TypeError, "numpy array"),
        ({"offsets": np.array([0, 1], dtype=np.int32)}, TypeError, "int64"),
        ({"offsets": _i64(1, 1)}, ValueError, "start with 0"),
        ({"offsets": _i64()}, ValueError, "start with 0"),
        ({"offsets": _i64(0, 1, 0, 1)}, ValueError, "not decrease"),
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
