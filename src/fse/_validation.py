"""Strict argument validation for the public entry points.

Only entry points whose own work dwarfs the validation cost use it: constructors, model
loading and the similarity searches. Per-element methods such as ``similarity`` or
``get_vector`` and the training data itself are checked explicitly instead.
"""

from pathlib import Path
from typing import Annotated, TypeAlias

import numpy as np
from gensim.models import FastText, Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from pydantic import BeforeValidator, ConfigDict, Field, validate_call

# strict: nothing is coerced, for example "4" is not accepted for an int.
# arbitrary_types_allowed: gensim and numpy objects are checked with isinstance.
validate_arguments = validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))

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
# Infinity would pass gt=0 and fail only after training, for example in int(cache_size_gb).
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
