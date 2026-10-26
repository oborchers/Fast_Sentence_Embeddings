"""Strict argument validation for the public entry points.

Only entry points whose own work dwarfs the validation cost use it: constructors, model
loading and the similarity searches. Per-element methods such as ``similarity`` or
``get_vector`` and the training data itself are checked explicitly instead.
"""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeAlias, TypeVar

import numpy as np
from gensim.models import FastText, Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from pydantic import BeforeValidator, ConfigDict, Field, validate_call

# strict: nothing is coerced, for example "4" is not accepted for an int.
# arbitrary_types_allowed: gensim and numpy objects are checked with isinstance.
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


#: Word vectors or a gensim model that holds them in ``wv``.
WordVectorModel: TypeAlias = KeyedVectors | Word2Vec | FastText

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
# Infinity would pass gt=0 and fail only later, for example as alpha in the SIF weights.
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
