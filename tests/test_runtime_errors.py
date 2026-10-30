"""Tests of fse._runtime._errors: the error of a run is that of its earliest failing row,
whatever the task boundaries and the workers, and only successful tasks are committed."""

import numpy as np
import pytest

from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline
from fse._runtime._errors import earliest, first_rows
from fse._text import StringSource
from fse._wordvectors import WordVectors

F32 = np.float32
# ok: an ordinary row; tiny: normalize@1 cannot scale it (its squares underflow); huge: two
# of them overflow while pooling; big: beyond the range of float16.
WORDS = ["ok", "tiny", "huge", "big"]
ROWS = np.array([[1, 2, 3, 4], [1e-30] * 4, [3e38] * 4, [1e5] * 4], F32)
NORMALIZED = Pipeline(
    EncodeSpec(WordVectors.from_arrays(ROWS, words=WORDS), np.ones(4, F32), (Normalize(),))
)
RAW = Pipeline(EncodeSpec(WordVectors.from_arrays(ROWS, words=WORDS), np.ones(4, F32), ()))
TOO_TINY = "tiny"
OVERFLOW = "huge huge"
TOO_BIG = "big"


class Raw(str):
    """A str whose own encode gives invalid UTF-8: the scan finds it (stage 1)."""

    def encode(self, *_args):
        return b"ok \xff"


class Recording:
    """A sink that records the rows committed to it."""

    def __init__(self, dim=4, dtype="float32"):
        self.dim, self.dtype, self.committed = dim, dtype, []

    def target(self, _first_row, n_rows):
        return np.empty((n_rows, self.dim), F32 if self.dtype == "float32" else np.float16)

    def commit(self, first_row, rows):
        self.committed.append((first_row, len(rows)))


def _error(pipeline, texts, **kwargs):
    with pytest.raises(ValueError, match=r"(row|sentence) \d+") as caught:
        pipeline.infer(StringSource(texts), **kwargs)
    return type(caught.value), str(caught.value)


def _oracle(pipeline, texts, **kwargs):
    """The error of the first row that fails when it is encoded alone, from the same
    bytes: the run of the texts up to it."""
    row = next(r for r, t in enumerate(texts) if _fails(pipeline, t, **kwargs))
    return _error(pipeline, texts[: row + 1], workers=1, **kwargs)


def _fails(pipeline, text, **kwargs):
    try:
        pipeline.infer(StringSource([text]), workers=1, **kwargs)
    except ValueError:
        return True
    return False


def _texts(n, placed):
    texts = ["ok " * (1 + i % 5) for i in range(n)]
    for row, text in placed.items():
        texts[row] = text
    return texts


CASES = [
    # (pipeline, the rows that fail and how, keyword arguments of infer)
    (NORMALIZED, {120: TOO_TINY, 2900: OVERFLOW}, {}),  # a late stage on the earlier row
    (NORMALIZED, {120: OVERFLOW, 2900: TOO_TINY}, {}),
    (NORMALIZED, {120: TOO_TINY, 121: OVERFLOW}, {}),  # neighbours, mostly in one task
    (RAW, {500: TOO_BIG, 2000: OVERFLOW}, {"dtype": "float16"}),  # storage against pooling
    (NORMALIZED, {700: TOO_TINY, 1800: Raw("x")}, {}),  # normalize against the scan
    (NORMALIZED, {700: Raw("x"), 1800: OVERFLOW}, {}),
]


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("workers", [1, 2, 10])
@pytest.mark.parametrize("chunk_bytes", [64, 4096, 1 << 30])
def test_the_earliest_failing_row_is_reported(case, workers, chunk_bytes):
    pipeline, placed, kwargs = CASES[case]
    texts = _texts(3000, placed)
    want = _oracle(pipeline, texts, **kwargs)
    assert str(min(placed)) in want[1]
    assert _error(pipeline, texts, workers=workers, chunk_bytes=chunk_bytes, **kwargs) == want


@pytest.mark.parametrize("workers", [1, 2, 10])
@pytest.mark.parametrize("chunk_bytes", [64, 4096, 1 << 30])
def test_only_successful_tasks_are_committed(workers, chunk_bytes):
    texts = _texts(3000, {1500: TOO_TINY, 2500: OVERFLOW})
    sink = Recording()
    with pytest.raises(ValueError, match="row 1500 cannot be normalized"):
        NORMALIZED.infer(StringSource(texts), workers=workers, chunk_bytes=chunk_bytes, sink=sink)
    rows = [first for first, _ in sink.committed]
    ends = [first + n for first, n in sink.committed]
    assert rows == [0, *ends[:-1]][: len(rows)]  # in order, each row once, from row 0
    assert not ends or ends[-1] <= 1500  # nothing of the failing task or after it


def test_earliest_finds_the_shortest_failing_prefix():
    calls = []

    def run(rows):
        calls.append(rows)
        if rows > 37:
            raise ValueError(f"fails from {rows}")

    with pytest.raises(ValueError, match="fails from 38"):
        earliest(run, 1000)
    assert calls[0] == 1000
    assert len(calls) <= 12
    earliest(lambda _rows: None, 5)


def test_earliest_passes_other_errors():
    def run(_rows):
        raise MemoryError("not a row")

    with pytest.raises(MemoryError):
        earliest(run, 10)


def test_first_rows():
    rows = np.arange(12).reshape(6, 2)
    assert first_rows(None, 3) is None
    assert first_rows(rows, 2).shape == (2, 2)
    codes, scales, norms = first_rows((rows, np.arange(6), None), 4)
    assert codes.shape == (4, 2)
    assert len(scales) == 4
    assert norms is None


@pytest.mark.parametrize("workers", [1, 2, 10])
@pytest.mark.parametrize("chunk_bytes", [64, 1 << 30])
def test_a_head_reports_the_earliest_failing_row(workers, chunk_bytes):
    from fse._contract import Link
    from fse._runtime._head import HeadSpec, make_head

    spec = RAW.spec
    head = HeadSpec(np.ones((2, 4), F32), np.zeros(2, F32), True, Link.SOFTMAX)
    run = make_head(spec, head, "cpu", own=Pipeline(spec)).infer
    texts = _texts(3000, {120: TOO_TINY, 2900: OVERFLOW})
    with pytest.raises(ValueError, match="row 120 cannot be normalized"):
        run(StringSource(texts), output="logits", workers=workers, chunk_bytes=chunk_bytes)


@pytest.mark.parametrize("workers", [1, 2, 10])
@pytest.mark.parametrize("chunk_bytes", [64, 4096, 1 << 30])
def test_a_file_reports_the_earliest_failing_row(tmp_path, workers, chunk_bytes):
    from fse._text_file import FileSource

    lines = [t.encode() for t in _texts(3000, {100: TOO_TINY, 2000: TOO_TINY})]
    lines[100], lines[2000] = OVERFLOW.encode(), b"ok \xff"  # pooling before the scan
    lines[2500] = b"bad \xfe"
    path = tmp_path / "lines.txt"
    path.write_bytes(b"\n".join(lines) + b"\n")
    with FileSource(path) as source, pytest.raises(ValueError, match="row 100 contains NaN"):
        NORMALIZED.infer(source, workers=workers, chunk_bytes=chunk_bytes)
    lines[100] = b"ok"
    path.write_bytes(b"\n".join(lines) + b"\n")
    with FileSource(path) as source, pytest.raises(UnicodeDecodeError, match="in sentence 2000 "):
        NORMALIZED.infer(source, workers=workers, chunk_bytes=chunk_bytes)
