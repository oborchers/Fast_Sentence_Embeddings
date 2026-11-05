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
    """The arguments of ``quotient64``: a ``weight`` (k, d) and a ``bias`` of k values, int
    or float arrays of at most 8 bytes per value (converted to float64 by the backend, in
    its floating-point environment), and a finite ``divisor`` greater than zero (a float)."""
    k = _matrix(weight, "weight", exact=False)
    _vector(bias, "bias", k, exact=False)
    if (
        isinstance(divisor, bool)
        or not isinstance(divisor, float | np.floating)
        or not np.isfinite(divisor)
        or divisor <= 0
    ):
        raise ValueError(f"divisor must be a finite float greater than zero, not {divisor!r}")


def _numbers(value: object, exact: bool) -> bool:
    if not isinstance(value, ndarray):
        return False
    if exact:
        return value.dtype == np.float64
    return value.dtype.kind in "fiu" and value.dtype != np.bool_ and value.dtype.itemsize <= 8


def _matrix(value: object, name: str, exact: bool = True) -> int:
    if not _numbers(value, exact) or value.ndim != 2:  # type: ignore[union-attr]
        raise ValueError(f"{name} must be a {'float64' if exact else 'numeric'} matrix")
    if not value.shape[0] or not value.shape[1]:
        raise ValueError(f"{name} must not be empty")
    return value.shape[0]


def _vector(value: object, name: str, length: int, exact: bool = True) -> None:
    if not _numbers(value, exact) or value.shape != (length,):  # type: ignore[union-attr]
        raise ValueError(
            f"{name} must be a {'float64' if exact else 'numeric'} vector of {length} values"
        )
