import numpy as np
import pydantic
import pytest

from fse._validation import (
    NonNegativeInt,
    PathLike,
    PositiveFloat,
    PositiveInt,
    Similarity,
    validate_arguments,
)


@validate_arguments
def _entry(
    count: PositiveInt,
    offset: NonNegativeInt = 0,
    alpha: PositiveFloat = 1.0,
    threshold: Similarity = 0.0,
    path: PathLike = "x",
):
    return count, offset, alpha, threshold, path


def test_valid_arguments_pass_unchanged_or_as_python_numbers():
    assert _entry(np.int64(3), offset=0, alpha=0.5, threshold=np.float32(0.5), path="a") == (
        3,
        0,
        0.5,
        0.5,
        "a",
    )
    assert type(_entry(np.int64(3))[0]) is int


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"count": True},
        {"count": "2"},
        {"count": 2.0},
        {"count": 1, "offset": -1},
        {"count": 1, "alpha": 0.0},
        {"count": 1, "alpha": float("inf")},
        {"count": 1, "threshold": 1.5},
        {"count": 1, "threshold": float("nan")},
        {"count": 1, "path": 3},
        {"count": 1, "unknown": 1},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(pydantic.ValidationError):
        _entry(**kwargs)


def test_threshold_input_shapes_and_array_identity():
    from pydantic import ConfigDict, TypeAdapter

    from fse._validation import ThresholdInput

    adapter = TypeAdapter(
        ThresholdInput, config=ConfigDict(strict=True, arbitrary_types_allowed=True)
    )
    for value in (
        None,
        0,
        1,
        0.4,
        np.float32(0.4),
        [0.4, 0.5],
        (0.4, 0.5),
        {"sport": 0.4},
        {np.int64(2): 0.4},
    ):
        adapter.validate_python(value)
    array = np.array([0.4, 0.5])
    assert adapter.validate_python(array) is array


@pytest.mark.parametrize(
    "value",
    [
        True,
        "0.4",
        [True, 0.5],
        {True: 0.4},
        {"sport": "0.4"},
        np.array([[0.4]]),
        np.array([np.nan]),
        np.array(["0.4"]),
        float("inf"),
        -1e-100,
        1 + 2**-30,
    ],
)
def test_threshold_input_rejects_coercion(value):
    from pydantic import ConfigDict, TypeAdapter, ValidationError

    from fse._validation import ThresholdInput

    adapter = TypeAdapter(
        ThresholdInput, config=ConfigDict(strict=True, arbitrary_types_allowed=True)
    )
    with pytest.raises(ValidationError):
        adapter.validate_python(value)


@pytest.mark.parametrize(
    "value", [[True, 1], [[0.5, np.bool_(True)]], np.ma.array([1], mask=False)]
)
def test_refuse_lost_input_rejects_boolean_elements_and_masks(value):
    from fse._validation import refuse_lost_input

    with pytest.raises(TypeError, match=r"bool|masked"):
        refuse_lost_input(value, "input")


@pytest.mark.parametrize("value", [[1, 0.5], (1, 2), np.array([1, 2], dtype=np.int32), []])
def test_refuse_lost_input_accepts_numbers(value):
    from fse._validation import refuse_lost_input

    assert refuse_lost_input(value, "input") is None


@pytest.mark.parametrize("value", [[np.ma.array([1, 2], mask=True)], [np.array([True]), [1.0]]])
def test_refuse_lost_input_checks_nested_arrays(value):
    from fse._validation import refuse_lost_input

    with pytest.raises(TypeError, match=r"masked|bool"):
        refuse_lost_input(value, "input")
