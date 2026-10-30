"""The attribution of the idle time of the workers (benchmarks/runtime_timeline_attr.py) on
events with hand-computed splits, the recording of the executor (runtime_timeline_probe.py)
and the helpers of the ablations and the report (PERFORMANCE.md, "Where the workers idle")."""

import importlib
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline, _executor, _pipeline
from fse._text import StringSource
from support.runtime import expected, texts, word_vectors

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"

pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "runtime_timeline_attr.py").exists(),
    reason="benchmarks/ is not part of this checkout",
)


@pytest.fixture
def attr(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("runtime_timeline_attr")


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("runtime_timeline_probe")


@pytest.fixture
def ablate(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("runtime_timeline_ablate")


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("runtime_timeline_run")


def _spec(steps):
    wv = word_vectors()
    return EncodeSpec(wv, np.ones(len(wv), np.float32), steps)


def _check(split, tasks):
    """The parts add up exactly: busy is the wall time of the tasks and busy plus every
    cause is the area."""
    assert split.busy == sum(t.end - t.start for t in tasks)
    assert split.busy + split.idle_total == split.area


def _two_workers_window_two(attr):
    """Task 0 is slow on thread A while task 1 (thread B) is done: with a full window the
    caller waits for task 0, B idles, and a third task still comes (time in ns)."""
    task = attr.Task
    return attr.Run(
        workers=2,
        window=2,
        inline=False,
        start=0,
        end=24,
        tasks=[task(0, 1, 1, 11, 3), task(1, 2, 3, 5, 4), task(2, 2, 16, 20, 5)],
        submits=[0, 2, 14],
        spans=[
            attr.Span("commit", 11, 12, 0),
            attr.Span("next", 12, 13),
            attr.Span("reserve", 13, 14),
            attr.Span("commit", 14, 15, 1),
            attr.Span("next", 15, 16),
            attr.Span("commit", 20, 21, 2),
        ],
        waits=[attr.Wait(4, 11, 2), attr.Wait(16, 20, 1)],
    )


def test_the_split_of_a_run_with_head_of_line_blocking(attr):
    run = _two_workers_window_two(attr)
    split = attr.attribute(run)
    assert (split.slots, split.start, split.end, split.area) == (2, 0, 20, 40)
    assert split.idle == {
        "tail": 4,  # thread A after the last task started
        "ramp": 4,  # A before its first task (1), B before its first (3)
        "hol": 6,  # B from 5 to 11: the caller waits for task 0, the queue is empty
        "producer": 2,  # the caller in next at 12 and 15..16 (A and B each at 12 to 13)
        "sink": 4,  # the caller in commit and reserve (11 to 12, 13 to 14) for both
        "handover": 4,  # 14 to 16: task 2 is queued, nobody has taken it
    }
    assert (split.before, split.after, split.after_commit) == (0, 4, 1)
    _check(split, run.tasks)


def test_a_wait_with_a_task_count_equal_to_the_window_is_no_head_of_line_blocking(attr):
    task = attr.Task
    run = attr.Run(
        workers=2,
        window=2,
        inline=False,
        start=0,
        end=13,
        tasks=[task(0, 1, 1, 10, 3), task(1, 2, 2, 4, 3)],
        submits=[0, 1],
        spans=[attr.Span("commit", 10, 11, 0), attr.Span("commit", 11, 12, 1)],
        waits=[attr.Wait(2, 10, 2), attr.Wait(11, 11, 1)],
    )
    split = attr.attribute(run)
    assert split.area == 20
    assert split.idle == {"tail": 6, "ramp": 3, "hol": 0, "producer": 0, "sink": 0, "handover": 0}
    _check(split, run.tasks)


def test_a_wait_with_a_window_that_is_not_full_is_no_head_of_line_blocking(attr):
    run = _two_workers_window_two(attr)
    run.waits[0] = attr.Wait(4, 11, 1)  # the window was not full
    split = attr.attribute(run)
    assert split.idle["hol"] == 0
    assert split.idle["handover"] == 4 + 6
    _check(split, run.tasks)


def test_the_run_without_a_later_task_is_no_head_of_line_blocking(attr):
    run = _two_workers_window_two(attr)
    run.submits = run.submits[:2]  # no task is submitted after the wait ended
    run.tasks = run.tasks[:2]
    split = attr.attribute(run)
    assert split.idle["hol"] == 0
    _check(split, run.tasks)


def test_a_slot_that_never_starts_idles_as_ramp_until_the_last_task_started(attr):
    task = attr.Task
    run = attr.Run(
        workers=4,
        window=8,
        inline=False,
        start=0,
        end=13,
        tasks=[task(0, 1, 1, 10, 3), task(1, 2, 2, 4, 3)],
        submits=[0, 1],
    )
    split = attr.attribute(run)
    assert (split.slots, split.area) == (4, 40)
    assert split.idle["ramp"] == 1 + 2 + 2 * 2  # two started slots and two that never start
    assert split.idle["tail"] == 6 + 2 * 8
    _check(split, run.tasks)


def test_ramp_goes_before_producer(attr):
    task = attr.Task
    run = attr.Run(
        workers=2,
        window=4,
        inline=False,
        start=0,
        end=12,
        tasks=[task(0, 1, 2, 10, 1), task(1, 2, 6, 11, 1)],
        submits=[0, 5],
        spans=[attr.Span("next", 2, 5), attr.Span("next", 5, 6)],
    )
    split = attr.attribute(run)
    assert split.idle["ramp"] == 2 + 6  # B waits for its thread while the caller is in next
    assert split.idle["producer"] == 0
    _check(split, run.tasks)


def test_an_inline_run_has_one_slot_and_no_ramp(attr):
    task, span = attr.Task, attr.Span
    run = attr.Run(
        workers=1,
        window=2,
        inline=True,
        start=0,
        end=12,
        tasks=[task(0, 9, 2, 5, 4), task(1, 9, 8, 10, 4)],
        spans=[
            span("next", 0, 1),
            span("reserve", 1, 2),
            span("commit", 5, 6, 0),
            span("next", 6, 7),
            span("reserve", 7, 8),
            span("commit", 10, 11, 1),
            span("next", 11, 12),
        ],
    )
    split = attr.attribute(run)
    assert (split.slots, split.area, split.busy) == (1, 10, 5)
    assert split.idle == {"tail": 0, "ramp": 0, "hol": 0, "producer": 2, "sink": 3, "handover": 0}
    assert (split.after, split.after_commit) == (2, 1)
    _check(split, run.tasks)


def test_a_run_without_a_task_has_no_area(attr):
    split = attr.attribute(attr.Run(workers=4, window=8, inline=False, start=5, end=9))
    assert (split.slots, split.area, split.busy, split.idle_total) == (0, 0, 0, 0)
    assert split.share("tail") == 0.0


def test_peak_in_flight_counts_reserved_and_not_committed(attr):
    span, task = attr.Span, attr.Task
    run = attr.Run(
        workers=2,
        window=2,
        inline=False,
        tasks=[task(0, 1, 2, 5, 3), task(1, 2, 3, 6, 4), task(2, 1, 13, 14, 5)],
        spans=[
            span("reserve", 0, 1),
            span("reserve", 1, 2),
            span("commit", 11, 12, 0),
            span("reserve", 13, 14),
            span("commit", 14, 15, 1),
            span("commit", 20, 21, 2),
        ],
    )
    assert attr.peak_in_flight(run) == (2, 9)


def test_thread_rates_name_the_rate_of_each_thread_only(attr):
    task = attr.Task
    run = attr.Run(
        workers=2,
        window=4,
        inline=False,
        tasks=[task(0, 1, 0, 10**9, 100), task(1, 2, 0, 2 * 10**9, 100), task(2, 1, 0, 10**9, 50)],
    )
    rates = attr.thread_rates(run)
    assert [r["tasks"] for r in rates] == [2, 1]
    assert rates[0]["rows_per_s"] == pytest.approx(75.0)
    assert rates[1]["rows_per_s"] == pytest.approx(50.0)


def _run_traced(probe, *, workers, count, window=None, pause=lambda _n: 0.002, reuse=None):
    rec = probe.Recorder()

    def work(item):
        time.sleep(pause(item))
        return item

    with probe.tracing(rec, reuse):
        _executor.run(
            iter(range(count)),
            work,
            lambda _item, _result: None,
            workers=workers,
            window=window,
        )
    return rec.runs[0]


def test_a_threaded_run_is_recorded_and_split_exactly(attr, probe):
    run = _run_traced(probe, workers=3, count=7)
    assert (run.workers, run.window, run.inline) == (3, 6, False)
    assert sorted(t.seq for t in run.tasks) == list(range(7))
    assert len(run.submits) == 7
    assert len(run.waits) >= 1
    split = attr.attribute(run)
    assert split.slots == 3
    _check(split, run.tasks)
    assert run.start <= split.start <= split.end <= run.end


def test_an_inline_run_is_recorded(attr, probe):
    run = _run_traced(probe, workers=1, count=4)
    assert run.inline
    assert len({t.thread for t in run.tasks}) == 1
    assert not run.submits
    split = attr.attribute(run)
    assert split.slots == 1
    _check(split, run.tasks)


def test_an_under_filled_run_has_slots_that_never_start(attr, probe):
    run = _run_traced(probe, workers=6, count=2, pause=lambda _n: 0.01)
    split = attr.attribute(run)
    assert split.slots == 6
    assert split.idle["ramp"] > 0
    _check(split, run.tasks)


def _slow_first(item):  # the pause of a task, by its number
    return 0.2 if item == 0 else 0.0


def test_head_of_line_blocking_shows_with_one_task_more_than_the_window(attr, probe):
    run = _run_traced(probe, workers=2, count=3, window=2, pause=_slow_first)
    split = attr.attribute(run)
    assert split.idle["hol"] > 0.05 * 1e9  # the second worker waits ~0.2 s for the window
    _check(split, run.tasks)


def test_no_head_of_line_blocking_when_the_tasks_fit_the_window(attr, probe):
    run = _run_traced(probe, workers=2, count=2, window=2, pause=_slow_first)
    split = attr.attribute(run)
    assert split.idle["hol"] == 0
    assert split.idle["tail"] > 0.05 * 1e9
    _check(split, run.tasks)


def test_the_patches_are_undone_and_an_error_still_leaves_its_run(probe):
    names = ("run", "ThreadPoolExecutor")
    before = [getattr(_executor, n) for n in names]
    commit, run_in_pipeline, infer = (
        _executor._Run._commit_oldest,
        _pipeline.run,
        Pipeline.infer,
    )
    rec = probe.Recorder()

    def boom(_item):
        raise RuntimeError("the task failed")

    with probe.tracing(rec), pytest.raises(RuntimeError, match="the task failed"):
        _executor.run(iter(range(4)), boom, lambda *_: None, workers=2)
    assert [getattr(_executor, n) for n in names] == before
    assert (_executor._Run._commit_oldest, _pipeline.run, Pipeline.infer) == (
        commit,
        run_in_pipeline,
        infer,
    )
    assert len(rec.runs) == 1
    assert rec.current is None


def test_a_reused_pool_keeps_its_threads_across_runs(probe):
    reuse = probe.Reuse()
    try:
        first = _run_traced(probe, workers=3, count=6, reuse=reuse)
        second = _run_traced(probe, workers=3, count=6, reuse=reuse)
        pool = reuse.pools[3]
        started = {t.ident for t in pool._threads}
        assert len(started) == 3
        assert {t.thread for t in first.tasks} | {t.thread for t in second.tasks} <= started
        assert any(t.name.startswith("fse-reused") for t in threading.enumerate())
    finally:
        reuse.close()
    assert not any(t.name.startswith("fse-reused") for t in threading.enumerate())


def test_tracing_a_real_pipeline_keeps_the_bits_and_the_sums_exact(attr, probe):
    spec = _spec((Normalize(),))
    sentences = texts(3000)
    pipeline = Pipeline(spec)
    plain = pipeline.infer(StringSource(sentences), workers=4)
    rec = probe.Recorder()
    with probe.tracing(rec):
        traced = pipeline.infer(StringSource(sentences), workers=4)
    assert traced.tobytes() == plain.tobytes() == expected(spec, sentences).tobytes()
    assert len(rec.runs) == 1 == len(rec.infers)
    run = rec.runs[0]
    assert sum(t.rows for t in run.tasks) == 3000
    split = attr.attribute(run)
    _check(split, run.tasks)
    assert rec.infers[0][0] <= run.start <= run.end <= rec.infers[0][1]


def test_the_window_of_an_ablation_changes_only_the_window(ablate, monkeypatch):
    monkeypatch.setattr(Pipeline, "plan_tasks", lambda *_args, **_kwargs: ("plan", 4, "stop"))
    with ablate.window_of(2):
        assert Pipeline.plan_tasks(None) == ("plan", 8, "stop")
    with ablate.window_of(None):
        assert Pipeline.plan_tasks(None)[1] == ablate.UNBOUNDED
    assert Pipeline.plan_tasks(None) == ("plan", 4, "stop")


def test_the_plan_of_an_ablation_is_the_plan_of_infer(ablate):
    spec = _spec(())
    sentences = texts(3000)
    pipeline = Pipeline(spec)
    rows = ablate.plan(pipeline, sentences, 4, None)
    assert sum(rows) == 3000
    smaller = ablate.plan(pipeline, sentences, 4, 1000)
    assert sum(smaller) == 3000
    assert len(smaller) > len(rows)
    assert "tasks, rows per task" in ablate._cuts(smaller)


def test_the_report_lines_of_a_summary(attr, report):
    run = _two_workers_window_two(attr)
    one = report.Pass(wall=30, before=2, run=run, split=attr.attribute(run))
    summary = report.summarize([one, one], 2)
    assert summary["exact"] is True
    assert summary["passes"] == 2
    assert sum(summary["idle_pct"].values()) == pytest.approx(summary["idle_pct_total"])
    assert summary["busy_pct"] + summary["idle_pct_total"] == pytest.approx(100.0)
    assert report.line(summary | {"discarded_blocks": 0}).startswith("      2")
    assert report.summarize([], 2) == {"workers": 2, "passes": 0}
    assert report.spread([2.0, 1.0, 0.0]) == 0.5
    assert report.spread([1.0]) == 0.0


@pytest.fixture
def cell():
    """A pipeline and texts above the inline limit, so the runs use threads."""
    pipeline = Pipeline(_spec((Normalize(),)))
    sentences = texts(6000)
    assert StringSource(sentences).nbytes > 1 << 16
    return pipeline, sentences


def test_a_series_of_a_cell_is_exact_and_threaded(report, cell):
    pipeline, sentences = cell
    result = report.series(pipeline, sentences, 4, 1, 1e9)
    assert result["exact"] is True
    assert result["passes"] == 5
    assert result["slots"] == 4
    assert result["tasks"] >= 4
    assert result["busy_pct"] + result["idle_pct_total"] == pytest.approx(100.0)


@pytest.mark.parametrize("which", ["chunk", "window", "pool", "sink"])
def test_every_ablation_runs_and_prints_its_variants(ablate, cell, which):
    pipeline, sentences = cell
    lines, result = getattr(ablate, which)(pipeline, sentences, 4, 1, 1e9)
    names = list(result["variants"])
    assert len(lines) >= 1 + len(names)
    assert all(result["variants"][n]["exact"] for n in names)
    assert all(result["variants"][n]["ratio_to_first"] for n in names)


def test_the_chunk_ablation_prints_the_real_task_counts(ablate, cell):
    pipeline, sentences = cell
    lines, result = ablate.chunk(pipeline, sentences, 4, 1, 1e9)
    counts = {n: v["tasks"] for n, v in result["variants"].items()}
    assert counts["8/worker"] > counts["1/worker"]
    assert any("tasks, rows per task" in line for line in lines)


def test_the_main_script_measures_the_overhead_of_the_tracing(monkeypatch, cell):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    main = importlib.import_module("runtime_timeline")
    pipeline, sentences = cell
    got = main.overhead(pipeline, sentences, 4, 2, 1, 1e9)
    assert got["rounds"] == 2
    assert len(got["ratios"]["traced"]) == 2
    text = "\n".join(main.overhead_lines(got))
    assert "traced / plain A" in text
    assert "plain B / plain A (noise)" in text
    args = main._parser().parse_args(
        ["ablate", "all", "--model", "potion-base-8M", "--corpus", "massive", "--sentences", "9"]
    )
    assert main._child_args(args, 4)[-2:] == ["--sentences", "9"]


def test_a_variant_without_a_kept_pass_prints_that(ablate):
    assert "no pass kept" in ablate.variant_line("x", {"workers": 8, "passes": 0})
