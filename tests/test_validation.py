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
from support.validation import validation_title


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
    def make(cls, _value: int) -> Self:
        return cls()

    @validate_arguments
    def copied(self, _value: int) -> Self:
        return type(self)()


class _SelfChild(_SelfValidated):
    pass


def test_self_return_annotations_keep_strict_validation_and_the_subclass():
    assert isinstance(_SelfChild.make(4), _SelfChild)
    assert isinstance(_SelfChild().copied(4), _SelfChild)
    for method in (_SelfChild.make, _SelfChild().copied):
        with pytest.raises(pydantic.ValidationError) as caught:
            method("4")
        assert caught.value.title == validation_title(method.__qualname__)


@pytest.mark.parametrize(
    "value", [[True, 1], [[0.5, np.bool_(True)]], np.ma.array([1], mask=False)]
)
def test_refuse_lost_input_rejects_boolean_elements_and_masks(value):
    from fse._validation import refuse_lost_input

    with pytest.raises(TypeError, match=r"bool|masked"):
        refuse_lost_input(value, "input")


@pytest.mark.parametrize(
    "value",
    [[1, 0.5], (1, 2), np.array([1, 2], dtype=np.int32), [np.float32(0.5), np.int64(2)], []],
)
def test_refuse_lost_input_accepts_numbers(value):
    from fse._validation import refuse_lost_input

    assert refuse_lost_input(value, "input") is None


@pytest.mark.parametrize("value", [[np.ma.array([1, 2], mask=True)], [np.array([True]), [1.0]]])
def test_refuse_lost_input_checks_nested_arrays(value):
    from fse._validation import refuse_lost_input

    with pytest.raises(TypeError, match=r"masked|bool"):
        refuse_lost_input(value, "input")


class _ArrayWithMetadata:
    """An array protocol object iterates metadata, as a DataFrame iterates columns."""

    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter([True, False])

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=True)


def test_refuse_lost_input_checks_array_values_instead_of_iterated_metadata():
    from fse._validation import refuse_lost_input

    assert refuse_lost_input(_ArrayWithMetadata([[1.0, 0.5]]), "weight") is None
    with pytest.raises(TypeError, match="bool"):
        refuse_lost_input(_ArrayWithMetadata([[True, 0.5]]), "weight")


def test_published_adapter_validator_compatibility():
    # Published 0.1.0 adapters import this until the next fse major release.
    from fse._validation import validate_arguments

    @validate_arguments
    def adapter(workers: int | None = None) -> int | None:
        return workers

    assert adapter(workers=1) == 1


def test_sink_target_limits_use_python_arithmetic_without_allocation():
    from fse._validation import target_rows

    maximum = int(np.iinfo(np.intp).max)
    assert target_rows(np.uint64(0), np.uint64(maximum // 12), 12) == (0, maximum // 12)
    with pytest.raises(ValueError, match=r"n_rows.*byte size"):
        target_rows(0, maximum // 12 + 1, 12)
    assert target_rows(maximum - 1, 1, 1) == (maximum - 1, 1)
    with pytest.raises(ValueError, match=r"n_rows.*target end"):
        target_rows(np.uint64(maximum), np.uint64(1), 1)
    assert target_rows(1, 1, 12, offset=maximum - 2) == (1, 1)
    with pytest.raises(ValueError, match=r"n_rows.*target end"):
        target_rows(1, 2, 12, offset=maximum - 2)
    with pytest.raises(ValueError, match=r"first_row.*store offset"):
        target_rows(maximum, 0, 12, offset=1)
