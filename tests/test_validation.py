from typing import Self

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


class _SelfValidated:
    @classmethod
    @validate_arguments
    def make(cls, value: int) -> Self:
        return cls()

    @validate_arguments
    def copied(self, value: int) -> Self:
        return type(self)()


class _SelfChild(_SelfValidated):
    pass


def test_self_return_annotations_keep_strict_validation_and_the_subclass():
    assert isinstance(_SelfChild.make(4), _SelfChild)
    assert isinstance(_SelfChild().copied(4), _SelfChild)
    for method in (_SelfChild.make, _SelfChild().copied):
        with pytest.raises(pydantic.ValidationError) as caught:
            method("4")
        assert caught.value.title == method.__qualname__
