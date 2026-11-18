"""Bounded selection and accounting for the CPU's per-call StringSource team."""

from __future__ import annotations

from itertools import chain
from typing import Any

from fse._text import StringSource

from ._sinks import ArrayRows

MAX_ROWS = 131_072
MAX_BYTES = 16 << 20
MAX_TASKS = 256


def eligible(
    pipeline, source, tasks, own, workers, pool, sink, base, lengths, collector, chunk_bytes
):
    """Decide before consuming any item of the plan's task iterator."""
    return (
        pool is None
        and sink is None
        and type(own) is ArrayRows
        and own.dtype == "float32"
        and type(source) is StringSource
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
    remaining = tasks.tasks
    if not eligible(
        pipeline, source, tasks, own, workers, pool, sink, base, lengths, collector, chunk_bytes
    ):
        return None, remaining
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
            return None, chain(buffered, (item,), remaining)
    else:
        return None, chain(buffered, remaining)
    replay = chain(buffered, _reraise(error))
    if not buffered:
        return None, replay
    try:
        team = pipeline.backend.string_team(
            pipeline.prepared, pipeline.lookup, source, buffered, own._out, workers
        )
    except (MemoryError, ValueError, TypeError, RuntimeError):
        return None, replay
    if not team.prepared:
        return None, replay
    return NativeCall(team, buffered, error), replay


class NativeCall:
    """A prepared team and the successful prefix it must book in task order."""

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
            work(reserve(self.tasks[failed_at]))
            raise RuntimeError("native task team and Python task error mismatch")
        if self.error is not None:
            raise self.error
