"""The conformance suite of a device backend (mlx) under its tolerance contract.

ARCHITECTURE.md, "Tolerance contract of device backends": every case of the bit-identical
suite, and cases with steps, within the bound of support.bound, on the GPU (a launch and
no host fallback, except for the cases with subnormal values, which fall back by design),
with the same bits on every run. The kernel of this backend keeps the cpu's order and fma,
so its rows also have the cpu's bits (a stronger property than the contract, checked here
and by tests/test_conformance.py). tests/test_conformance_device_meta.py shows that the
comparison catches the mistakes the contract names.
"""

import hashlib
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from fse import _reference
from fse._backends._mlx import _kernels, _sources
from fse._contract import Normalize, Project, RemoveComponents
from fse._runtime import EncodeSpec, Pipeline
from fse._text import StringSource
from fse._wordvectors import WordVectors
from support.bound import assert_within_bound
from support.cases import CASES, random_case
from support.conformance import (
    F32,
    TINY,
    Case,
    Contract,
    _rng,
    assert_bits_equal,
    mlx_module,
    pool_on_device,
)
from support.runtime import expected, source, texts, word_vectors

# The cases the GPU cannot compute because a value of the arithmetic is or may be
# subnormal: the host computes them.
SUBNORMAL = {
    ("float32", "subnormal rows"),
    ("float32", "subnormal products"),
    ("float32", "subnormal to normal"),
    ("int8", "subnormal products"),
    ("int8", "subnormal to normal"),
    ("int8", "int8 tiny scales"),
    ("float32", "tiny oov weight"),
    ("float32", "cancels to subnormal"),
    # The product 2**-195 of 2**-70 and 2**-125 may underflow (their exponent fields sum
    # to less than 128), so the word is flagged although its fma sum is normal.
    ("float32", "subnormal product"),
}


@pytest.fixture
def mlx():
    return Contract("mlx").backend  # skips where it cannot run


@pytest.mark.parametrize("index", range(len(CASES)), ids=[f"{c.dtype}-{c.name}" for c in CASES])
def test_every_case_is_within_the_bound(mlx, index):
    case = CASES[index]
    out, fell_back, launched = pool_on_device(mlx, case)
    again, _, _ = pool_on_device(mlx, case)
    assert_bits_equal(again, out, f"{case.name}, run 2")
    assert_within_bound(out, case, name=case.name)
    if (case.dtype, case.name) in SUBNORMAL:
        assert fell_back
    elif case.sentences:
        assert launched, case.name  # on the GPU, not on the host
        assert not fell_back, case.name
    assert_bits_equal(out, _reference.pool_weighted_mean(case.batch, *case.args()), case.name)


def _basis(dim, k, seed):
    q = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, k)))[0]
    return np.ascontiguousarray(q.T, dtype=F32)


STEPS = {
    "normalize": (Normalize(),),
    "remove": (RemoveComponents(_basis(12, 2, 1), np.array([1.0, 0.5], F32)),),
    "all": (
        RemoveComponents(_basis(12, 2, 1), np.array([1.0, 0.5], F32)),
        Project(_basis(12, 5, 2)),
        Normalize(),
    ),
}


@pytest.mark.parametrize("steps", list(STEPS))
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("fasttext", [False, True], ids=["plain", "fasttext"])
def test_steps_are_within_the_bound(mlx, steps, dtype, fasttext):
    case = random_case("steps", dtype, 40, dim=12, n=30, ft=fasttext)
    out, fell_back, launched = pool_on_device(mlx, case, STEPS[steps])
    assert launched
    assert not fell_back
    assert_within_bound(out, case, STEPS[steps], "steps")


def _check_fallbacks(backend):
    """A host fallback exactly for the cases with subnormal values."""
    fell_back = set()
    for case in CASES:
        _, count, _ = pool_on_device(backend, case)
        if count:
            fell_back.add((case.dtype, case.name))
    assert fell_back == SUBNORMAL


def test_only_the_subnormal_cases_fall_back(mlx):
    _check_fallbacks(mlx)


def test_a_backend_that_always_falls_back_is_caught(mlx, monkeypatch):
    """Its rows are right (the host computes them), so only the count catches it."""
    real = _kernels.read
    monkeypatch.setattr(_kernels, "read", lambda *a: real(*a) | _sources.SUBNORMAL)
    with pytest.raises(AssertionError):
        _check_fallbacks(mlx)


# --- Determinism: the bits of a row depend on its own inputs only ---------------------------

MIXED = Case(
    "mixed",
    "float32",
    np.vstack([_rng(5).standard_normal((6, 8)).astype(F32), np.full((1, 8), TINY * 3, F32)]),
    _rng(6).random(7).astype(F32),
    ((0, 1, 2), (6, 1), (3, 4), (), (5, 0, 0), (6,), (2, 2, 4)),
)
FLAGGED = [1, 5]  # the sentences with the subnormal word 6


def _parts(case, cuts):
    """``case`` in the batches between ``cuts``, as cases of their own."""
    bounds = [0, *cuts, len(case.sentences)]
    return [
        Case(case.name, case.dtype, case.base, case.weights, case.sentences[b:e], first_row=b)
        for b, e in itertools.pairwise(bounds)
        if e > b
    ]


@pytest.mark.parametrize("cuts", [(), (1,), (1, 2), (5,), (2, 4, 6), tuple(range(1, 7))])
def test_a_flagged_row_changes_no_other_row(mlx, cuts):
    """Whether the flagged rows share a task with others or not: the unflagged rows keep
    the device's bits, the flagged ones the cpu's."""
    whole, fell_back, _ = pool_on_device(mlx, MIXED)
    assert fell_back
    parts = [pool_on_device(mlx, part)[0] for part in _parts(MIXED, cuts)]
    assert_bits_equal(np.concatenate(parts), whole, f"cuts {cuts}")
    want = _reference.pool_weighted_mean(MIXED.batch, *MIXED.args())
    assert_bits_equal(whole[FLAGGED], want[FLAGGED], "flagged")


@pytest.mark.parametrize("group_tasks", [1, 2, 8])
def test_the_rows_do_not_depend_on_the_device_groups(mlx, monkeypatch, group_tasks):
    """Tasks from several threads evaluated in groups of 1, 2 or 8 tasks."""
    monkeypatch.setattr(mlx.device, "group_tasks", group_tasks)
    parts = _parts(MIXED, tuple(range(1, 7)))
    want, _, _ = pool_on_device(mlx, MIXED)
    with ThreadPoolExecutor(len(parts)) as pool:
        got = list(pool.map(lambda part: pool_on_device(mlx, part)[0], parts))
    assert_bits_equal(np.concatenate(got), want, f"groups of {group_tasks}")


def test_a_fresh_process_gives_the_same_bits(mlx):
    case = random_case("fresh", "float16", 9, dim=300, n=40, length=30, ft=True)
    out, _, launched = pool_on_device(mlx, case)
    assert launched
    code = (
        "import hashlib, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from support.cases import random_case\n"
        "from support.conformance import Contract, pool_on_device\n"
        "case = random_case('fresh', 'float16', 9, dim=300, n=40, length=30, ft=True)\n"
        "out, _, _ = pool_on_device(Contract('mlx').backend, case)\n"
        "print(hashlib.sha256(out.tobytes()).hexdigest())\n"
    )
    done = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert done.stdout.strip() == hashlib.sha256(out.tobytes()).hexdigest()


def _subnormal_word_vectors():
    wv = word_vectors()
    rows = np.array(wv.stored_vectors)
    rows[5, 3] = TINY * 7
    return WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(len(rows))])


@pytest.mark.parametrize(("workers", "chunk_bytes"), [(1, 1 << 20), (3, 512), (3, 4096)])
@pytest.mark.parametrize("kind", ["strings", "tokens"])
@pytest.mark.usefixtures("mlx")
def test_workers_tasks_and_inputs_give_the_same_bits(tmp_path, workers, chunk_bytes, kind):
    """The runtime with a word whose subnormal value sends some rows to the host: every
    worker count, task size and input class with the same tokens gives the same bits."""
    spec = EncodeSpec(_subnormal_word_vectors(), np.ones(200, F32), (Normalize(),))
    if kind == "tokens":
        spec = EncodeSpec(spec.word_vectors, spec.weights, spec.steps, "external")
    sentences = texts(600)
    pipeline = Pipeline(spec, "mlx")
    got = pipeline.infer(
        source(kind, sentences, tmp_path), workers=workers, chunk_bytes=chunk_bytes
    )
    assert got.tobytes() == expected(spec, sentences).tobytes()
    assert sum(mlx_module("fse._backends._mlx._backend").fallbacks(pipeline.prepared).values())


@pytest.mark.usefixtures("mlx")
@pytest.mark.parametrize("store", ["float16", "int8"])
def test_stage_5_quantizes_the_device_rows_exactly(monkeypatch, store):
    """The codes and scales are the cpu quantization of the device's own float32 rows
    (captured as they are stored), which here also have the cpu's bits."""
    spec = EncodeSpec(_subnormal_word_vectors(), np.ones(200, F32), (Normalize(),))
    sentences = texts(800)
    pipeline = Pipeline(spec, "mlx")
    captured = {}
    real = pipeline.backend.store

    def store_rows(rows, codes, scales, first_row):
        captured[first_row] = rows.copy()
        real(rows, codes, scales, first_row)

    monkeypatch.setattr(pipeline.backend, "store", store_rows)
    got = pipeline.infer(StringSource(sentences), workers=3, chunk_bytes=1024, dtype=store)
    rows = np.concatenate([captured[k] for k in sorted(captured)])
    codes, scales = _reference.quantize_rows(rows, store)
    if store == "int8":
        assert_bits_equal(got[0], codes, "codes")
        assert_bits_equal(got[1], scales, "scales")
    else:
        assert_bits_equal(got, codes, "codes")
    assert_bits_equal(rows, expected(spec, sentences), "rows")
