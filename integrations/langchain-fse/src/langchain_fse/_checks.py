"""Package-owned validation. Keep identical across the three adapters."""

import os
from pathlib import Path

import numpy as np

from fse import Encoder
from fse.typing import ModelRef, WorkerCount


def model_reference(value: ModelRef) -> ModelRef:
    """Preserve strings as Hub references and path objects as local paths."""
    if isinstance(value, str):
        return value
    if not isinstance(value, os.PathLike):
        raise TypeError("model must be a string or a text PathLike")
    path = os.fspath(value)
    if not isinstance(path, str):
        raise TypeError("model PathLike must return str, not bytes")
    return Path(path).absolute()


def worker_count(value: WorkerCount) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"workers must be a positive int, not {value!r}")
    return int(value)


def configuration(
    model: ModelRef,
    encoder: Encoder | None,
    dtype: str,
    workers: WorkerCount,
    load_kwargs: dict[str, object] | None,
) -> tuple[ModelRef, int | None, dict[str, object]]:
    model = model_reference(model)
    if encoder is not None and not isinstance(encoder, Encoder):
        raise TypeError("encoder must be an Encoder or None")
    if not isinstance(dtype, str):
        raise TypeError("dtype must be a string")
    if dtype not in ("float32", "float16"):
        raise ValueError("dtype must be float32 or float16 for float-list embeddings")
    workers = worker_count(workers)
    if load_kwargs is not None and (
        not isinstance(load_kwargs, dict) or any(not isinstance(k, str) for k in load_kwargs)
    ):
        raise TypeError("load_kwargs must be a dict with string keys or None")
    if encoder is not None and load_kwargs:
        raise ValueError("load_kwargs cannot be used with an existing encoder")
    return model, workers, dict(load_kwargs or {})
