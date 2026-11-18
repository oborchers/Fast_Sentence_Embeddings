"""Bounded selection and accounting for the CPU's per-call StringSource team."""

from __future__ import annotations

from itertools import chain
from typing import Any

from fse._text import CopiedStrings, StringSource

from ._sinks import ArrayRows
from ._worker import WorkerLocal

MAX_ROWS = 131_072
MAX_BYTES = 16 << 20
MAX_TASKS = 256


def _scope(
    pipeline, source, tasks, workers, pool, sink, base, lengths, collector, chunk_bytes, dtype
):
    return (
        pool is None
        and sink is None
        and dtype == "float32"
        and type(source) in (StringSource, CopiedStrings)
        and source._verified
        and base == 0
        and lengths is None
        and collector is None
        and chunk_bytes is None
        and not tasks.inline
        and workers > 1
        and pipeline.backend.name == "cpu"
        and pipeline._text_rows
        and callable(getattr(pipeline.backend, "string_tasks", None))
        and callable(getattr(pipeline.backend, "string_team", None))
        and tasks.rows is not None
        and tasks.rows <= MAX_ROWS
        and source.nbytes <= MAX_BYTES
    )


def eligible(
    pipeline, source, tasks, own, workers, pool, sink, base, lengths, collector, chunk_bytes
):
    """Decide before consuming any item of the plan's task iterator."""
    return type(own) is ArrayRows and _scope(
        pipeline,
        source,
        tasks,
        workers,
        pool,
        sink,
        base,
        lengths,
        collector,
        chunk_bytes,
        own.dtype,
    )


def prepare(
    pipeline,
    source,
    tasks,
    workers,
    window,
    dtype,
    sink,
    pool,
    base,
    lengths,
    collector,
    chunk_bytes,
    output_factory,
):
    """Keep old-path logging before setup; delay candidates to select executor metadata."""
    log_args = (source, tasks.rows, tasks.inline, workers, window, dtype, sink)
    logged = not _scope(
        pipeline, source, tasks, workers, pool, sink, base, lengths, collector, chunk_bytes, dtype
    )
    if logged:
        pipeline._planned(*log_args)
    try:
        own = output_factory(tasks.rows, pipeline.output_dim, dtype) if sink is None else None
        active_sink = own if sink is None else sink
        local = WorkerLocal(lengths)
        work = pipeline._task_work(source, type(own) is ArrayRows, local, base, collector)
        internal = own if work is not None else None
        team, remaining = select(
            pipeline, source, tasks, own, workers, pool, sink, base, lengths, collector, chunk_bytes
        )
        if not logged:
            logged = True
            pipeline._planned(*log_args, executor=getattr(team, "executor", None))
        return own, active_sink, local, work, internal, team, remaining
    except BaseException:
        if not logged:
            pipeline._planned(*log_args)
        raise


def _append(buffered, item):
    """A separate operation so tests can fail buffering rather than source iteration."""
    buffered.append(item)


def _reraise(error):
    if error is not None:
        raise error
    yield from ()


def select(
    pipeline, source, tasks, own, workers, pool, sink, base, lengths, collector, chunk_bytes
):
    """Return a prepared call, or an iterator preserving all buffered tasks and errors."""
    if not eligible(
        pipeline, source, tasks, own, workers, pool, sink, base, lengths, collector, chunk_bytes
    ):
        return None, tasks.tasks
    remaining = tasks.tasks
    buffered, error, replay = _buffer(remaining)
    if buffered is None or not buffered:
        return None, replay
    try:
        team = pipeline.backend.string_team(
            pipeline.prepared, pipeline.lookup, source, buffered, own._out, workers
        )
    except (MemoryError, ValueError, TypeError):
        return None, replay
    if not team.prepared:
        return None, replay
    return NativeCall(team, buffered, error), replay


def _buffer(remaining):
    buffered: list[Any] = []
    error = None
    for _ in range(MAX_TASKS + 1):
        try:
            item = next(remaining)
        except StopIteration:
            break
        except Exception as found:
            error = found
            break
        try:
            _append(buffered, item)
        except MemoryError:
            return None, None, chain(buffered, (item,), remaining)
    else:
        return None, None, chain(buffered, remaining)
    replay = chain(buffered, _reraise(error))
    return buffered, error, replay


def execute(team, own, done, work, reserve, commit, remaining, *, runner, **options):
    """Execute the selected route with the existing executor's untouched options."""
    if team is None:
        runner(remaining, work, commit, reserve=reserve, **options)
    else:
        team.execute(own, done, work, reserve)


class NativeCall:
    """A prepared team and the successful prefix it must book in task order."""

    executor = "native"

    def __init__(self, team, tasks, error):
        self.team, self.tasks, self.error = team, tasks, error

    def execute(self, own, done, work, reserve):
        failed_at, no_memory = self.team.run()
        for task in self.tasks[:failed_at]:
            own.next_row += task.n_rows
            done[0] += task.n_rows
            done[1] += 1
        if failed_at < len(self.tasks):
            if no_memory:
                raise MemoryError("could not grow the buffers of the Workspace")
            item = reserve(self.tasks[failed_at])
            state = self.team.retry_state
            work(item if state is None else (*item, state))
            raise RuntimeError("native task team and Python task error mismatch")
        if self.error is not None:
            raise self.error
