"""The checks of the float64 arithmetic with which a head is imported (every backend)."""

from __future__ import annotations

import numpy as np
from numpy import ndarray


def check_affine64_input(weight: object, bias: object, factor: object, shift: object) -> None:
    """The arguments of ``affine64``: a float64 ``weight`` (k, d), and float64 ``bias``,
    ``factor`` and ``shift`` of k values each."""
    k = _matrix(weight, "weight")
    for name, value in (("bias", bias), ("factor", factor), ("shift", shift)):
        _vector(value, name, k)


def check_quotient64_input(weight: object, bias: object, divisor: object) -> None:
    """The arguments of ``quotient64``: a float64 ``weight`` (k, d), a float64 ``bias`` of k
    values and a finite ``divisor`` greater than zero (a float)."""
    k = _matrix(weight, "weight")
    _vector(bias, "bias", k)
    if (
        isinstance(divisor, bool)
        or not isinstance(divisor, float | np.floating)
        or not np.isfinite(divisor)
        or divisor <= 0
    ):
        raise ValueError(f"divisor must be a finite float greater than zero, not {divisor!r}")


def _matrix(value: object, name: str) -> int:
    if not isinstance(value, ndarray) or value.dtype != np.float64 or value.ndim != 2:
        raise ValueError(f"{name} must be a float64 matrix")
    if not value.shape[0] or not value.shape[1]:
        raise ValueError(f"{name} must not be empty")
    return value.shape[0]


def _vector(value: object, name: str, length: int) -> None:
    if not isinstance(value, ndarray) or value.dtype != np.float64 or value.shape != (length,):
        raise ValueError(f"{name} must be a float64 vector of {length} values")
