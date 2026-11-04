"""Strict argument validation for the public entry points.

Only entry points whose own work dwarfs the validation cost use it: constructors, model
loading and the similarity searches. Per-element methods such as ``similarity`` or
``get_vector`` and the training data itself are checked explicitly instead.
"""

import functools
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, TypeAlias, TypeVar

import numpy as np
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    validate_call,
)

# strict: nothing is coerced, for example "4" is not accepted for an int.
# arbitrary_types_allowed: gensim and numpy objects are checked with isinstance.
# fse._validation imports no gensim, so 2.0 modules can use it without gensim installed.
_validate = validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))

F = TypeVar("F", bound=Callable[..., Any])


def validate_arguments(func: F) -> F:
    """Validate the arguments of ``func`` strictly with pydantic.

    The plain function wrapper binds to instances like any method. pydantic before 2.6
    caches its own wrapper in the instance ``__dict__`` when a decorated method is
    accessed, and that wrapper cannot be pickled, which breaks saving models.
    """
    validated = _validate(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return validated(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


#: A file system path given as str or Path.
PathLike: TypeAlias = str | Path


def _numpy_integer_to_int(value: object) -> object:
    """Accept numpy integer scalars such as ``np.int64(4)``, which strict mode rejects."""
    if isinstance(value, np.integer):
        return int(value)
    return value


# bool stays an error: np.bool_ is no np.integer, and strict mode rejects a Python bool.
PositiveInt = Annotated[int, BeforeValidator(_numpy_integer_to_int), Field(gt=0)]
NonNegativeInt = Annotated[int, BeforeValidator(_numpy_integer_to_int), Field(ge=0)]
# Infinity would pass gt=0 and fail only later, in the arithmetic it is given to.
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def _numpy_floating_to_float(value: object) -> object:
    """Accept numpy float scalars such as ``np.float32(0.5)``, which strict mode rejects."""
    if isinstance(value, np.floating):
        return float(value)
    return value


#: A cosine similarity threshold.
Similarity = Annotated[
    float, BeforeValidator(_numpy_floating_to_float), Field(ge=-1, le=1, allow_inf_nan=False)
]


def _probability(value: object) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise ValueError("a probability must be a number, not a bool or string")
    if not 0 <= value <= 1:
        raise ValueError("a probability must be finite and in [0, 1]")
    return float(value)


def _threshold_label(value: object) -> object:
    return int(value) if isinstance(value, np.integer) else value


def _threshold_vector(value: np.ndarray) -> np.ndarray:
    if value.ndim != 1:
        raise ValueError("thresholds must be a one-dimensional vector")
    for item in value:
        _probability(item)
    return value


_Probability = Annotated[
    float, BeforeValidator(_probability), Field(ge=0, le=1, allow_inf_nan=False)
]
_ThresholdLabel = Annotated[StrictStr | StrictInt, BeforeValidator(_threshold_label)]
_ThresholdVector = Annotated[np.ndarray, AfterValidator(_threshold_vector)]
#: Validated cutoffs, not a runtime object to construct. Arrays are checked without copying.
ThresholdInput: TypeAlias = (
    _Probability
    | list[_Probability]
    | tuple[_Probability, ...]
    | Mapping[_ThresholdLabel, _Probability]
    | _ThresholdVector
    | None
)
