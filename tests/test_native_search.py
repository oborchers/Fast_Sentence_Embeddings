"""The search in the kernel: inverse norms, cosines and scores against fse._reference."""

import numpy as np
import pytest

from fse import _native, _reference

F32 = np.float32
ISAS = _native.available_isas()
DTYPES = ["float32", "float16", "int8"]


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _stored(dtype, n, dim, seed):
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((n, dim)).astype(F32)
    if n > 2 and dim > 1:
        rows[1] = 0  # a zero row
        rows[2, : dim // 2] = -0.0
        if dtype == "float32":
            rows[2, 0] = F32(1e-40)  # a subnormal beside normal values
    return _reference.quantize_rows(rows, dtype)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [1, 15, 16, 17, 33, 300])
def test_inverse_norms_cosines_and_scores_give_the_reference_bits(isa, dtype, dim):
    stored, scales = _stored(dtype, 23, dim, dim)
    inverse = _native.inverse_norms(stored)
    np.testing.assert_array_equal(bits(inverse), bits(_reference.inverse_norms(stored)))
    rng = np.random.default_rng(dim + 1)
    for q in (1, 2, 3, 5, 17):
        queries = rng.standard_normal((q, dim)).astype(F32)
        np.testing.assert_array_equal(
            bits(_native.cosines(stored, inverse, queries, isa=isa)),
            bits(_reference.cosines(stored, inverse, queries)),
        )
        biases = rng.standard_normal(q).astype(F32)
        np.testing.assert_array_equal(
            bits(_native.scores(stored, scales, queries, biases, isa=isa)),
            bits(_reference.scores(stored, scales, queries, biases)),
        )


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("env", [{"flush": True}, {"round_up": True}])
def test_the_search_runs_in_the_environment_of_the_kernel(isa, env):
    stored, scales = _stored("float16", 9, 33, 5)
    queries = np.random.default_rng(6).standard_normal((3, 33)).astype(F32) * F32(1e-20)
    weights = queries * F32(1e-19)  # subnormal, computed before the environment changes
    inverse = _reference.inverse_norms(stored)
    want = [
        bits(inverse),
        bits(_reference.cosines(stored, inverse, queries)),
        bits(_reference.scores(stored, scales, weights)),
    ]
    saved = _native._test_fpenv_set(**env)
    try:
        got = [
            bits(_native.inverse_norms(stored)),
            bits(_native.cosines(stored, inverse, queries, isa=isa)),
            bits(_native.scores(stored, scales, weights, isa=isa)),
        ]
    finally:
        _native._test_fpenv_restore(saved)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)


def test_errors_name_the_query_or_vector_and_the_row_as_the_reference():
    big = F32(3e38)
    stored = np.array([[1, 0], [big, big], [big, big]], F32)
    inverse = np.ones(3, F32)
    for fn in (_native.cosines, _reference.cosines):
        with pytest.raises(ValueError, match="query 1 is the zero vector"):
            fn(stored, inverse, np.array([[1, 0], [0, 0]], F32))
        with pytest.raises(ValueError, match=r"query 0: row 0 cannot be normalized"):
            fn(stored, inverse, np.array([[1e-30, 0], [0, 0.5]], F32))
        with pytest.raises(ValueError, match=r"query 1: the cosine of row 1 is not finite"):
            fn(stored, inverse, np.array([[1, 0], [1, 1]], F32))
    for fn in (_native.scores, _reference.scores):
        with pytest.raises(ValueError, match=r"weights 1: the score of row 1 overflows"):
            fn(stored, None, np.array([[1, 0], [big, big]], F32))
    tiny = np.array([[1e-45, 0]], F32)
    for fn in (_native.inverse_norms, _reference.inverse_norms):
        with pytest.raises(ValueError, match="the inverse norm of row 0 is beyond float32"):
            fn(tiny)


def test_arguments_are_checked():
    stored = np.ones((2, 3), F32)
    inverse = np.ones(2, F32)
    with pytest.raises(ValueError, match="queries must be a matrix"):
        _native.cosines(stored, inverse, np.ones(3, F32))
    with pytest.raises(ValueError, match="queries must hold 3 values per vector"):
        _native.cosines(stored, inverse, np.ones((1, 2), F32))
    with pytest.raises(ValueError, match="inverse must be a float32 vector of 2 values"):
        _native.cosines(stored, np.ones(3, F32), np.ones((1, 3), F32))
    with pytest.raises(ValueError, match="scales are given exactly for int8 rows"):
        _native.scores(stored, inverse, np.ones((1, 3), F32))
    codes = np.ones((2, 3), np.int8)
    with pytest.raises(ValueError, match="scales are given exactly for int8 rows"):
        _native.scores(codes, None, np.ones((1, 3), F32))


def test_empty_inputs():
    stored = np.zeros((0, 4), F32)
    assert _native.inverse_norms(stored).shape == (0,)
    assert _native.cosines(stored, np.zeros(0, F32), np.ones((2, 4), F32)).shape == (0, 2)
    assert _native.scores(stored, None, np.ones((3, 4), F32)).shape == (0, 3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [1, 3, 300])
@pytest.mark.parametrize("n", [*range(1, 9), *range(16, 25)])
def test_inverse_norms_of_interleaved_rows_give_the_reference_bits(dtype, dim, n):
    # The kernel sums 8 rows at a time: every rest of 0 to 7 rows, alone and after full
    # groups, gives the bits of one row at a time.
    stored = _stored(dtype, n, dim, seed=n)
    codes = stored[0] if isinstance(stored, tuple) else stored
    got = _native.inverse_norms(codes, 5)
    np.testing.assert_array_equal(bits(got), bits(_reference.inverse_norms(codes, 5)))


@pytest.mark.parametrize("bad", [0, 7, 8, 11, 16, 18])
def test_an_inverse_norm_beyond_float32_names_its_row_at_every_group_position(bad):
    rows = np.ones((19, 4), dtype=F32)  # two full groups and a rest of 3
    rows[bad] = F32(1e-40)  # subnormal: its inverse norm overflows float32
    rows[bad + 1 :][:1] = F32(1e-40)  # a later bad row loses
    for first_row in (0, 100):
        for fn in (_native.inverse_norms, _reference.inverse_norms):
            with pytest.raises(ValueError, match=f"row {first_row + bad} is beyond float32"):
                fn(rows, first_row)
