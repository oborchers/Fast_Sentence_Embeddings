"""fse._backends._mlx._kernels without a GPU: the arrays of a task, the split of a batch
beyond a device group, the launch shape, the flags of the rows."""

import numpy as np
import pytest

from fse._backends import PoolInput
from fse._backends._mlx import _kernels, _sources
from fse._contract import OOV_TOKEN, Batch

F32 = np.float32
TABLE = _kernels.Table(
    (), dim=3, n_words=5, n_buckets=4, int8=False, fasttext=True, oov_weight=True
)
U = OOV_TOKEN


def _batch(sentences, ngrams=None, first_row=3):
    offsets = np.cumsum([0, *map(len, sentences)]).astype(np.int64)
    tokens = np.array([t for s in sentences for t in s], np.uint32)
    if ngrams is None:
        return Batch(first_row, offsets, tokens)
    ngram_offsets = np.cumsum([0, *map(len, ngrams)]).astype(np.int64)
    flat = np.array([b for n in ngrams for b in n], np.uint32)
    return Batch(first_row, offsets, tokens, ngram_offsets, flat)


def test_a_task_numbers_the_first_unknown_word_of_every_sentence():
    batch = _batch(((U, 1), (), (2,), (U, U, 4), (U,)), ((0, 1), (), (3,), (2, 2)))
    task = _kernels.task(batch, TABLE)
    assert task is not None
    assert task.unknown
    assert (task.tokens, task.n_sentences, task.first_row) == (7 + 5, 5, 3)  # with buckets
    assert task.arrays[2].tolist() == [0, 1, 1, 1, 3]
    assert [a.dtype for a in task.arrays] == [np.uint32] * 5


def test_a_task_without_unknown_words_reads_no_ngrams():
    task = _kernels.task(_batch(((0, 4), (1,))), TABLE)
    assert task is not None
    assert not task.unknown
    assert [len(a) for a in task.arrays[2:]] == [1, 1, 1]
    assert task.arrays[1].dtype == np.uint32


@pytest.mark.parametrize(
    ("batch", "table"),
    [
        (_batch(((0, 5),)), TABLE),  # a word beyond the rows
        (_batch(((U,),), ((4,),)), TABLE),  # a bucket beyond the ngram rows
        (_batch(((U,),), ((0,),)), _kernels.Table((), 3, 5, 0, False, False, False)),
        (_batch(((U,),), ((0,),)), _kernels.Table((), 3, 5, 4, False, True, False)),
    ],
    ids=["word", "bucket", "no-fasttext", "no-oov-weight"],
)
def test_a_batch_the_cpu_refuses_is_no_task(batch, table):
    assert _kernels.task(batch, table) is None


def test_a_task_beyond_uint32_offsets_is_refused(monkeypatch):
    monkeypatch.setattr(_kernels, "LIMIT", 3)
    with pytest.raises(ValueError, match="at most 3 tokens"):
        _kernels.task(_batch(((0, 1), (2, 3))), TABLE)


def test_the_sizes_count_the_buckets_of_each_sentence():
    batch = _batch(((1, U), (2, 3, 4), (), (U,), (U, 1, 2)), ((1, 2, 3), (), (4,)))
    assert _kernels.sizes(batch).tolist() == [5, 3, 0, 1, 4]
    assert _kernels.size(batch) == 13
    assert _kernels.sizes(_batch(((1, 2), ()))).tolist() == [2, 0]


def _parts(batch, budget, limit=100):
    out = []
    for begin, end, part in _kernels.split(batch, budget, limit):
        if part is None:
            out.append((begin, end, None))
            continue
        flat = None if part.ngrams is None else part.ngrams.tolist()
        out.append((begin, end, part.first_row, part.tokens.tolist(), flat))
    return out


def test_a_batch_splits_into_runs_that_fit_and_long_sentences_alone():
    batch = _batch(((1, U), (2, 3, 4), (), (U,), (U, 1, 2)), ((1, 2, 3), (), (4,)))
    assert _parts(batch, 5) == [
        (0, 1, 3, [1, U], [1, 2, 3]),
        (1, 4, 4, [2, 3, 4, U], []),
        (4, 5, 7, [U, 1, 2], [4]),
    ]
    assert _parts(batch, 4, 4) == [
        (0, 1, None),  # 5 contributions, beyond the limit: the host's
        (1, 4, 4, [2, 3, 4, U], []),
        (4, 5, 7, [U, 1, 2], [4]),
    ]
    # beyond the budget but within the limit: alone, on the device
    assert _parts(batch, 2, 100) == [
        (0, 1, 3, [1, U], [1, 2, 3]),
        (1, 2, 4, [2, 3, 4], []),
        (2, 4, 5, [U], []),
        (4, 5, 7, [U, 1, 2], [4]),
    ]
    assert _parts(batch, 100) == [(0, 5, 3, [1, U, 2, 3, 4, U, U, 1, 2], [1, 2, 3, 4])]


def test_the_parts_of_a_split_are_batches_of_their_own():
    batch = _batch(((1, U), (2, 3, 4), (), (U,), (U, 1, 2)), ((1, 2, 3), (), (4,)))
    for begin, end, part in _kernels.split(batch, 4, 4):
        if part is not None:
            assert part.n_sentences == end - begin
            assert part.ngram_offsets[0] == 0


@pytest.mark.parametrize(
    ("dim", "vec", "rows", "want"),
    [
        (300, 4, 10, ((96, 10, 1), (96, 2, 1), 3)),
        (300, 4, 1, ((96, 1, 1), (96, 1, 1), 3)),
        (301, 1, 7, ((512, 7, 1), (256, 1, 1), 16)),
        (8, 4, 50, ((32, 50, 1), (32, 8, 1), 1)),
        (65536, 4, 3, ((16384, 3, 1), (256, 1, 1), 512)),
    ],
)
def test_the_launch_shape(dim, vec, rows, want):
    """Threads of a row padded to SIMD groups, at most 256 per group along a row."""
    assert _kernels.shape(dim, vec, rows) == want


def test_a_table_takes_four_values_per_thread_for_aligned_rows():
    assert _kernels.Table((), 300, 1, 0, False, False, False).vec == 4
    assert _kernels.Table((), 301, 1, 0, False, False, False).vec == 1


def test_the_flags_of_a_sentence_combine_its_words():
    """OR over the words of a sentence; the bounds copied."""
    long, nonfinite = _sources.LONG, _sources.NONFINITE
    words = np.array([[long, 0], [0, 0], [nonfinite, long]], np.uint32)
    launched = (np.zeros((3, 1), F32), words, np.ones((3, 2), F32))
    _, flags, bounds = _kernels.view(launched)
    assert flags.tolist() == [long, 0, nonfinite | long]
    assert bounds.tolist() == [[1, 1]] * 3
    assert _kernels.flagged(flags).tolist() == [True, False, True]


def test_the_reason_of_a_fallback():
    assert _kernels.reason(_sources.NONFINITE) == "nonfinite"
    assert _kernels.reason(_sources.LONG | _sources.NONFINITE) == "long"
    assert _kernels.reason(0) == "limit"


TINY = F32(np.finfo(F32).smallest_subnormal)


def _pool(**changes):
    arrays = {"vectors": np.ones((2, 3), F32), "weights": np.ones(2, F32)}
    arrays.update(changes)
    return PoolInput(**arrays)


@pytest.mark.parametrize(
    ("pool", "oov", "refused"),
    [
        (_pool(), None, False),
        (_pool(weights=np.array([1.0, TINY], F32)), None, True),
        (_pool(vectors=np.array([[1.0, TINY, 0.0], [1, 1, 1]], F32)), None, True),
        (_pool(vectors=np.full((2, 3), 2.0**-24, np.float16)), None, False),  # widens normal
        (_pool(weights=np.array([0.0, 1.0], F32)), None, False),
        # int8: a subnormal scale, and weight * scale whose exponent fields sum to 127
        (_pool(vectors=np.ones((2, 3), np.int8), scales=np.array([1, TINY], F32)), None, True),
        (
            _pool(
                vectors=np.ones((2, 3), np.int8),
                weights=np.array([2.0**-61, 1], F32),
                scales=np.array([2.0**-66, 1], F32),
            ),
            None,
            True,
        ),
        (
            _pool(
                vectors=np.ones((2, 3), np.int8),
                weights=np.array([2.0**-60, 1], F32),
                scales=np.array([2.0**-66, 1], F32),
            ),
            None,
            False,
        ),
        # FastText: (1 / m) * oov_weight for m below 2**24, times the ngram scale
        (_pool(ngram_vectors=np.ones((4, 3), F32)), np.array([2.0**-102], F32), False),
        (_pool(ngram_vectors=np.ones((4, 3), F32)), np.array([2.0**-103], F32), True),
        (_pool(ngram_vectors=np.ones((4, 3), F32)), np.array([TINY], F32), True),
        (_pool(ngram_vectors=np.array([[TINY, 0, 0]] * 4, F32)), np.ones(1, F32), True),
        # normal inputs with subnormal factors: 2**-63 * 2**-64, and (1 / 2) * 2**-126
        (
            _pool(
                vectors=np.ones((2, 3), np.int8),
                weights=np.array([2.0**-63, 1], F32),
                scales=np.array([2.0**-64, 1], F32),
            ),
            None,
            True,
        ),
        (_pool(ngram_vectors=np.full((4, 3), 2.0**126, F32)), np.array([2.0**-126], F32), True),
        # int8 ngrams: 2**-24 (the smallest 1 / m) times 2**-103 is subnormal, 2**-102 not
        (
            _pool(
                vectors=np.ones((2, 3), np.int8),
                scales=np.ones(2, F32),
                ngram_vectors=np.ones((4, 3), np.int8),
                ngram_scales=np.array([1, 1, 1, 2.0**-103], F32),
            ),
            np.ones(1, F32),
            True,
        ),
        (
            _pool(
                vectors=np.ones((2, 3), np.int8),
                scales=np.ones(2, F32),
                ngram_vectors=np.ones((4, 3), np.int8),
                ngram_scales=np.array([1, 1, 1, 2.0**-102], F32),
            ),
            np.ones(1, F32),
            False,
        ),
    ],
    ids=[
        "normal",
        "weight",
        "row value",
        "float16",
        "zero weight",
        "int8 scale",
        "int8 factor",
        "int8 factor at 2**-126",
        "oov at the limit",
        "oov below",
        "oov subnormal",
        "ngram row",
        "int8 factor of normal inputs",
        "ngram factor of normal inputs",
        "int8 ngram factor",
        "int8 ngram factor at 2**-126",
    ],
)
def test_subnormal_inputs_are_refused_at_prepare(pool, oov, refused):
    assert _kernels.subnormal_input(pool, oov) is refused


def test_the_check_reads_the_rows_in_parts(monkeypatch):
    monkeypatch.setattr(_kernels, "CHECK_VALUES", 4)
    rows = np.ones((3, 3), F32)
    assert not _kernels.subnormal_input(_pool(vectors=rows), None)
    rows[2, 2] = TINY
    assert _kernels.subnormal_input(_pool(vectors=rows), None)
