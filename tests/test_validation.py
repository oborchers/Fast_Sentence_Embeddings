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
