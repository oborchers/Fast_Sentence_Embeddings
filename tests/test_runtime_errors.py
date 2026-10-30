"""Tests of fse._runtime._errors: the error of a run is that of its earliest failing row,
whatever the task boundaries and the workers, and only successful tasks are committed."""

import re

import numpy as np
import pytest

from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline
from fse._runtime._errors import earliest, first_rows
from fse._text import StringSource
from fse._wordvectors import WordVectors
from support.arrow_c import Producer

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


class Column:
    """An Arrow string column with nulls, exported through the C stream interface without
    pyarrow (``support.arrow_c``): ``None`` is a null."""

    def __init__(self, texts):
        self.producer = Producer()
        self.strings = [b"" if t is None else t.encode() for t in texts]
        self.nulls = [i for i, t in enumerate(texts) if t is None]

    def __arrow_c_stream__(self, requested_schema=None):
        p = self.producer
        return p.stream(p.schema(b"u"), [p.strings(b"u", self.strings, nulls=self.nulls)])


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


def _isolated(pipeline, text, row, **kwargs):
    """The error of ``text`` encoded alone as row ``row`` of a run (into a sink numbered
    from it), so neither the rows around it nor the task boundaries take part; the byte of
    the source, which depends on the rows before it, is left out."""
    dtype = kwargs.get("dtype", "float32")
    with pytest.raises(ValueError, match=rf"(row|sentence) {row}\b") as caught:
        pipeline.infer(StringSource([text]), workers=1, sink=Recording(dtype=dtype), first_row=row)
    return type(caught.value), _bytes_out(str(caught.value))


def _bytes_out(message):
    return re.sub(r"\(byte \d+ of the source\)", "(byte of the source)", message)


def _texts(n, placed):
    texts = ["ok " * (1 + i % 5) for i in range(n)]
    for row, text in placed.items():
        texts[row] = text
    return texts


NORMALIZE = r"row {} cannot be normalized"
POOLING = r"row {} contains NaN or infinite values"
FLOAT16 = r"row {} has values beyond the float16 range"
UTF8 = r"in sentence {} \(byte \d+ of the source\)"
CASES = [
    # (pipeline, the rows that fail and how, keyword arguments of infer, the error of the
    # earliest row by construction)
    (NORMALIZED, {120: TOO_TINY, 2900: OVERFLOW}, {}, NORMALIZE),  # a late stage, earlier row
    (NORMALIZED, {120: OVERFLOW, 2900: TOO_TINY}, {}, POOLING),
    (NORMALIZED, {120: TOO_TINY, 121: OVERFLOW}, {}, NORMALIZE),  # neighbours, mostly one task
    (RAW, {500: TOO_BIG, 2000: OVERFLOW}, {"dtype": "float16"}, FLOAT16),  # storage, pooling
    (NORMALIZED, {700: TOO_TINY, 1800: Raw("x")}, {}, NORMALIZE),  # normalize against the scan
    (NORMALIZED, {700: Raw("x"), 1800: OVERFLOW}, {}, UTF8),
]


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("workers", [1, 2, 10])
@pytest.mark.parametrize("chunk_bytes", [64, 4096, 1 << 30])
def test_the_earliest_failing_row_is_reported(case, workers, chunk_bytes):
    pipeline, placed, kwargs, pattern = CASES[case]
    row = min(placed)
    texts = _texts(3000, placed)
    kind, message = _error(pipeline, texts, workers=workers, chunk_bytes=chunk_bytes, **kwargs)
    assert re.search(pattern.format(row), message)  # the row of the run, named explicitly
    assert (kind, _bytes_out(message)) == _isolated(pipeline, texts[row], row, **kwargs)


def test_errors_of_a_second_batch_name_the_rows_of_the_run():
    """A batch from row 1000 of a run: the scan, the length rule of the tokenizers and an
    Arrow null all name the row of the run."""
    from fse import _native
    from fse._text import ArrowSource

    texts = _texts(50, {5: Raw("x")})
    with pytest.raises(UnicodeDecodeError, match=r"in sentence 1005 \(byte \d+ of the source\)"):
        NORMALIZED.infer(StringSource(texts), sink=Recording(), first_row=1000)
    previous = _native._test_sentence_limit(64)
    try:
        long = _texts(50, {7: "ok " * 40})
        with pytest.raises(ValueError, match=r"sentence 1007 has 120 bytes"):
            NORMALIZED.infer(StringSource(long), sink=Recording(), first_row=1000)
    finally:
        _native._test_sentence_limit(previous)
    column = Column(["ok", "ok", "ok", None, "ok"])
    with pytest.raises(ValueError, match=r"row 1003 of the Arrow column is null"):
        NORMALIZED.infer(ArrowSource(column), sink=Recording(), first_row=1000)


@pytest.mark.parametrize("workers", [1, 10])
def test_an_arrow_null_comes_after_an_earlier_pooling_error(workers):
    from fse._text import ArrowSource

    with pytest.raises(ValueError, match="row 1 contains NaN"):
        NORMALIZED.infer(ArrowSource(Column(["ok", OVERFLOW, None])), workers=workers)
    with pytest.raises(ValueError, match="row 1 of the Arrow column is null"):
        NORMALIZED.infer(ArrowSource(Column(["ok", None, OVERFLOW])), workers=workers)


def test_for_one_row_the_head_comes_before_the_storage_of_its_vectors():
    """Row 3 has values beyond float16 (its vectors cannot be stored, stage 5) and logits
    that overflow (stage 4b): the head's error is raised."""
    from fse._contract import Link
    from fse._runtime._head import HeadSpec, make_head

    spec = RAW.spec
    head = HeadSpec(np.full((2, 4), 1e34, F32), np.zeros(2, F32), False, Link.SOFTMAX)
    pipeline = make_head(spec, head, "cpu", own=Pipeline(spec), with_vectors=True)
    texts = ["ok", "ok", "ok", TOO_BIG, "ok"]
    with pytest.raises(ValueError, match="the logits of row 3 are not finite"):
        pipeline.infer(StringSource(texts), output="logits", vector_dtype="float16")


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


def test_a_failed_run_holds_no_memory_while_its_prefixes_run():
    """An error only in the last row: the error of the whole run is kept while the shorter
    runs make their own arrays, but not the arrays of the failed run."""
    import tracemalloc

    rows = 64

    def run(n):
        batch = np.ones(n * (1 << 15))  # 256 KiB per row
        if n == rows:
            raise ValueError(f"row {n - 1} fails")
        del batch

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="row 63 fails"):
            earliest(run, rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.5 * rows * (1 << 18)  # one run's array, not the failed one's as well


@pytest.mark.parametrize("sink", [False, True])
def test_errors_of_a_stream_name_the_rows_of_the_run(sink):
    """Several collections of 3 sentences, from row 1000 of a run: a sentence that is no
    str and one with a lone surrogate name their rows of the run."""
    from fse._stream import StreamSource

    kwargs = {"sink": Recording()} if sink else {}
    for bad, error, pattern in [
        (None, TypeError, r"string 1007 is a NoneType"),
        ("ok \ud800", UnicodeEncodeError, r"in string 1007"),
    ]:
        texts = ["ok"] * 7 + [bad] + ["ok"] * 3
        with pytest.raises(error, match=pattern):
            NORMALIZED.infer(StreamSource(iter(texts), chunk_rows=3), first_row=1000, **kwargs)
