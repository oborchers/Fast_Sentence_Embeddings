"""Shared scan runners preserve chunks, thread ownership and error order."""

import threading

import pytest

from fse import Pool
from fse._runtime._executor import run
from fse._text import ArrowSource, StringSource, _scan
from fse._text_file import FileSource
from support.pool import no_fse_threads_left  # noqa: F401


@pytest.fixture(params=["file", "encoded", "arrow", "verified"])
def source(request, tmp_path):
    text = "w1 w2 " * 40
    rows = [text] * 5000  # above the scan threshold, not just the encoding threshold
    if request.param == "file":
        path = tmp_path / "text"
        path.write_text("\n".join(rows))
        with FileSource(path) as src:
            yield src
    elif request.param == "arrow":
        pa = pytest.importorskip("pyarrow")
        yield ArrowSource(pa.array(rows))
    elif request.param == "encoded":
        yield StringSource.encoded([row.encode() for row in rows])
    else:
        yield StringSource(rows)


def test_runner_preserves_chunks_and_uses_only_pool_threads(source, monkeypatch):
    wanted, error = source.scan_until_error(workers=3, chunk_bytes=4096)
    assert error is None
    calls, names = [], []
    real = source._scanner

    def scanner(*args):
        fn = real(*args)

        def scan(span):
            names.append(threading.current_thread().name)
            return fn(span)

        return scan

    def no_executor(**_kwargs):
        raise AssertionError("the scan made its own executor")

    monkeypatch.setattr(source, "_scanner", scanner)
    monkeypatch.setattr(_scan, "ThreadPoolExecutor", no_executor)
    with Pool(3, max_tasks=1) as pool:

        def runner(fn, spans):
            calls.append(1)
            results = []
            run(spans, fn, lambda _, r: results.append(r), workers=3, pool=pool)
            return results

        got, error = source.scan_until_error(workers=3, chunk_bytes=4096, run_ranges=runner)
        assert got == wanted
        assert error is None
        if source._verified:
            assert not calls
            assert not names
        else:
            assert calls == [1]
            assert names
            assert all(name.startswith("fse-worker") for name in names)
        assert not pool.closed


@pytest.mark.parametrize("offsets", [False, True])
def test_an_earlier_invalid_sentence_wins_over_a_later_task_exception(
    tmp_path, monkeypatch, offsets
):
    path = tmp_path / "bad"
    path.write_bytes(b"\xff\n" + b"w1\n" * 600_000)
    src = StringSource.encoded([b"\xff", *([b"w1"] * 600_000)]) if offsets else FileSource(path)
    real = src._scanner
    later = []

    def scanner(*args):
        fn = real(*args)

        def scan(span):
            if span[0] > 0:
                later.append(span)
                raise MemoryError("later range")
            return fn(span)

        return scan

    monkeypatch.setattr(src, "_scanner", scanner)
    try:
        with Pool(2) as pool:

            def runner(fn, spans):
                results = []
                run(spans, fn, lambda _, r: results.append(r), workers=2, pool=pool)
                return results

            scan, error = src.scan_until_error(workers=2, run_ranges=runner)
            assert later
            assert scan.sentences == 0
            assert isinstance(error, UnicodeDecodeError)
            assert "sentence 0" in str(error)
            assert pool.try_acquire()
            pool.release()
    finally:
        if isinstance(src, FileSource):
            src.close()


def test_without_a_runner_large_unverified_sources_keep_their_own_threads(monkeypatch):
    src = StringSource.encoded([b"w1 " * 100] * 5000)
    real = src._scanner
    names = []

    def scanner(*args):
        fn = real(*args)

        def scan(span):
            names.append(threading.current_thread().name)
            return fn(span)

        return scan

    monkeypatch.setattr(src, "_scanner", scanner)
    scan, error = src.scan_until_error(workers=2)
    assert error is None
    assert scan.sentences == 5000
    assert names
    assert all(name.startswith("fse-text") for name in names)


def test_runner_task_errors_are_raised_in_range_order(monkeypatch):
    src = StringSource.encoded([b"w1 " * 100] * 5000)

    def scanner(*_args):
        def scan(span):
            raise MemoryError(str(span[0]))

        return scan

    monkeypatch.setattr(src, "_scanner", scanner)
    with Pool(2) as pool:

        def runner(fn, spans):
            results = []
            run(spans, fn, lambda _, r: results.append(r), workers=2, pool=pool)
            return results

        with pytest.raises(MemoryError, match=r"^0$"):
            src.scan_until_error(workers=2, run_ranges=runner)
        assert pool.try_acquire()
        pool.release()
