"""The settings, the collector and the report of the length rule, and the pipeline that
fills them: the same report for any workers, chunk size and backend, and rows that equal
the rows without a report bit for bit."""

import dataclasses
import threading

import numpy as np
import pytest

import fse
from fse._contract import TruncationError
from fse._encoder._sources import opened
from fse._runtime import LengthCollector, LengthReport, LengthSettings
from support import lengths as sup

TEXTS = sup.TEXTS


def bits(rows):
    return np.ascontiguousarray(rows).view(np.uint8).tobytes()


def pipeline(backend="cpu", **rule):
    return sup.encoder(**rule)._pipeline(backend)


def run(pipe, texts, settings, **keywords):
    """The rows and the report of a pipeline run."""
    collector = LengthCollector(settings, 0, (5, None))
    with opened(texts, "wordpiece@1") as source:
        rows = pipe.infer(source, lengths=settings, collector=collector, **keywords)
    return rows, collector.report()


# --- the settings ---------------------------------------------------------------------------


def test_the_defaults_make_no_settings():
    assert LengthSettings.of(None, False, "keep", "m") is None


def test_the_settings_follow_the_keywords():
    one = LengthSettings.of("summary", False, "keep", "m")
    assert (one.watch, one.store_rows, one.raise_cut, one.row_bytes) == (1, False, False, 0)
    two = LengthSettings.of("per_row", True, "raise", "m")
    assert (two.watch, two.store_rows, two.raise_cut, two.row_bytes) == (2, True, True, 33)
    assert LengthSettings.of("per_row", False, "keep", "m").row_bytes == 17
    raise_only = LengthSettings.of(None, False, "raise", "m")
    assert (raise_only.lengths, raise_only.raise_cut, raise_only.row_bytes) == (None, True, 0)


@pytest.mark.parametrize(
    ("arguments", "error", "text"),
    [
        ((1, False, "keep"), TypeError, "m: lengths must be None, 'summary' or 'per_row', not int"),
        (
            ("rows", False, "keep"),
            ValueError,
            "m: lengths must be None, 'summary' or 'per_row', not 'rows'",
        ),
        (("summary", 1, "keep"), TypeError, "m: count_truncated must be a bool, not int"),
        (
            ("summary", False, None),
            TypeError,
            "m: on_truncation must be 'keep' or 'raise', not NoneType",
        ),
        (
            ("summary", False, "ignore"),
            ValueError,
            "m: on_truncation must be 'keep' or 'raise', not 'ignore'",
        ),
        (
            (None, True, "keep"),
            ValueError,
            "m: count_truncated needs lengths='summary' or 'per_row'",
        ),
    ],
)
def test_the_settings_are_checked(arguments, error, text):
    with pytest.raises(error) as caught:
        LengthSettings.of(*arguments, "m")
    assert str(caught.value) == text


# --- the report -----------------------------------------------------------------------------


def test_a_report_is_frozen_and_read_only():
    _, report = run(pipeline(), TEXTS, LengthSettings.of("per_row", True, "keep", "m"))
    assert isinstance(report, LengthReport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.n_rows = 1  # type: ignore[misc]
    for name in ("tokens", "empty", "char_truncated", "truncated_tokens", "truncated_chars"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(report, name)[0] = 1
    assert fse.LengthReport is LengthReport


def test_the_empty_run_has_an_empty_report():
    _, report = run(pipeline(), [], LengthSettings.of("per_row", True, "keep", "m"))
    assert report.n_rows == 0
    assert report.tokens.tolist() == []
    assert report.truncated_rows().tolist() == []
    assert report.n_truncated == 0


# --- the pipeline ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("chunk_bytes", [None, 300])
def test_the_report_does_not_depend_on_workers_chunks_or_the_backend(backend, workers, chunk_bytes):
    pipe = pipeline(backend)
    settings = LengthSettings.of("per_row", True, "keep", "m")
    with opened(TEXTS, "wordpiece@1") as source:
        plain = pipe.infer(source, workers=workers, chunk_bytes=chunk_bytes)
    rows, report = run(pipe, TEXTS, settings, workers=workers, chunk_bytes=chunk_bytes)
    assert bits(rows) == bits(plain)
    sup.check_report(report, sup.expected(TEXTS), count=True)


def test_first_row_numbers_the_report_and_the_errors():
    pipe = pipeline()
    settings = LengthSettings.of("per_row", False, "keep", "m")
    collector = LengthCollector(settings, 40, (5, None))
    with opened(TEXTS, "wordpiece@1") as source:
        pipe.infer(source, lengths=settings, collector=collector, first_row=40)
    report = collector.report()
    assert report.first_row == 40
    assert report.truncated_rows()[0] >= 40
    raising = LengthSettings.of(None, False, "raise", "m")
    with pytest.raises(TruncationError) as caught, opened(TEXTS, "wordpiece@1") as source:
        pipe.infer(source, lengths=raising, first_row=40)
    want = sup.expected(TEXTS)
    first = next(k for k, w in enumerate(want) if w["token_truncated"])
    assert caught.value.row == 40 + first


def test_a_task_of_empty_texts_reports_nothing_of_the_task_before():
    pipe = pipeline()
    settings = LengthSettings.of("per_row", False, "keep", "m")
    texts = [*TEXTS[-70:], *[""] * 400, "a"]
    _, report = run(pipe, texts, settings, workers=1, chunk_bytes=200)
    sup.check_report(report, sup.expected(texts))


def test_parallel_calls_with_other_settings_keep_their_own():
    pipe = pipeline()
    results = {}

    def call(name, level, count):
        settings = LengthSettings.of(level, count, "keep", "m")
        results[name] = run(pipe, TEXTS, settings, workers=2)[1]

    threads = [
        threading.Thread(target=call, args=(name, level, count))
        for name, level, count in (
            ("a", "summary", False),
            ("b", "per_row", True),
            ("c", "per_row", False),
        )
        for _ in range(1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    want = sup.expected(TEXTS)
    sup.check_report(results["a"], want, per_row=False)
    sup.check_report(results["b"], want, count=True)
    sup.check_report(results["c"], want)


def test_the_next_run_starts_with_a_clean_record():
    pipe = pipeline()
    settings = LengthSettings.of("summary", False, "keep", "m")
    first = run(pipe, TEXTS, settings)[1]
    again = run(pipe, TEXTS, settings)[1]
    assert first.n_truncated == again.n_truncated > 0
    assert run(pipe, TEXTS[:3], settings)[1].n_rows == 3


def test_the_record_of_a_failed_run_does_not_leak_into_the_next():
    pipe = pipeline()
    raising = LengthSettings.of("summary", False, "raise", "m")
    with pytest.raises(TruncationError):
        run(pipe, TEXTS, raising)
    plain = LengthSettings.of("summary", False, "keep", "m")
    sup.check_report(run(pipe, TEXTS, plain)[1], sup.expected(TEXTS), per_row=False)


def test_a_sink_run_reports_too():
    from fse import MemorySink

    pipe = pipeline()
    settings = LengthSettings.of("per_row", False, "keep", "m")
    collector = LengthCollector(settings, 0, (5, None))
    sink = MemorySink(pipe.output_dim)
    with opened(TEXTS, "wordpiece@1") as source:
        pipe.infer(source, sink=sink, lengths=settings, collector=collector)
    sup.check_report(collector.report(), sup.expected(TEXTS))
