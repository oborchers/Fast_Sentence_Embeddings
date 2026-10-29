"""fse._backends._mlx._kernels without a GPU: the arrays of a task, the split of a batch
beyond a device group, the launch shape, the flags of the rows and the canary's inputs."""

import numpy as np
import pytest

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


def _parts(batch, limit):
    out = []
    for begin, end, part in _kernels.split(batch, limit):
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
    assert _parts(batch, 4) == [
        (0, 1, None),
        (1, 4, 4, [2, 3, 4, U], []),
        (4, 5, 7, [U, 1, 2], [4]),
    ]
    assert _parts(batch, 100) == [(0, 5, 3, [1, U, 2, 3, 4, U, U, 1, 2], [1, 2, 3, 4])]


def test_the_parts_of_a_split_are_batches_of_their_own():
    batch = _batch(((1, U), (2, 3, 4), (), (U,), (U, 1, 2)), ((1, 2, 3), (), (4,)))
    for begin, end, part in _kernels.split(batch, 4):
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
    """OR over the words, TINY only where every word has it."""
    tiny, sub = _sources.TINY, _sources.SUBNORMAL
    words = np.array([[tiny, tiny], [tiny, 0], [sub, tiny], [0, 0]], np.uint32)
    launched = (np.zeros((4, 1), F32), words)
    flags = _kernels.read(launched, np.empty((4, 1), F32))
    assert flags.tolist() == [tiny, 0, sub, 0]


def test_the_flagged_rows():
    """Before normalize also rows near its limits, but not an empty row (exact zero)."""
    task = _kernels.task(_batch(((1,), (), (2,), (3,), (4,))), TABLE)
    s, t, h, n, long = (
        _sources.SUBNORMAL,
        _sources.TINY,
        _sources.HUGE,
        _sources.NONFINITE,
        _sources.LONG,
    )
    flags = np.array([s | t, t, h, n | long, 0], np.uint32)
    assert _kernels.flagged(flags, task, False).tolist() == [True, False, False, True, False]
    assert _kernels.flagged(flags, task, True).tolist() == [True, False, True, True, False]
    flags[4] = t
    assert _kernels.flagged(flags, task, True)[4]


def test_the_reason_of_a_fallback():
    table = _kernels.Table((), 3, 5, 0, False, False, False, np.array([0, 0, 1, 0, 0], bool))
    plain = _kernels.task(_batch(((0, 1),)), table)
    flagged_word = _kernels.task(_batch(((2, 1),)), table)
    assert _kernels.reason(_sources.SUBNORMAL, plain, table) == "underflow"
    assert _kernels.reason(_sources.NONFINITE, flagged_word, table) == "underflow"
    assert _kernels.reason(_sources.NONFINITE, plain, table) == "nonfinite"
    assert _kernels.reason(_sources.LONG | _sources.NONFINITE, plain, table) == "long"
    assert _kernels.reason(_sources.HUGE, plain, table) == "limit"


def test_the_canary_inputs_are_the_values_it_names():
    f = F32(1 + 2**-12)
    rows = np.array(
        [f, -(f * f), 1 + 2**-23, 2.0**-149, 1.25 * 2.0**-126, -(2.0**-126), 3.4028235e38],
        F32,
    )
    rows = np.append(rows, -rows[2])
    assert _kernels.CANARY_ROWS.tobytes() == rows.tobytes()
    assert _kernels.CANARY_WEIGHTS.tobytes() == np.array([f, *[1] * 7], F32).tobytes()
    assert np.array([2.0**-24], np.float16).view(np.uint16)[0] == _kernels.CANARY_HALF[0]
    assert np.array([2.0**-24], F32).view(np.uint32)[0] == _kernels.CANARY_HALF[1]
