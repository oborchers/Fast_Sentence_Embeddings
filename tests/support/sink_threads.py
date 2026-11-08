"""The writer threads of the sinks (``fse-shard-*``) as the process sees them, and the
check that a test leaves none behind that it started."""

import threading

from support.waiting import until

PREFIX = "fse-shard"


def sink_threads() -> set[threading.Thread]:
    """The live writer threads of all sinks of this process (the thread objects)."""
    return {t for t in threading.enumerate() if t.name.startswith(PREFIX)}


def left_since(before: set[threading.Thread], timeout: float = 10) -> list[threading.Thread]:
    """The writer threads that did not exist in ``before`` and still live after at most
    ``timeout`` seconds (a closed sink's threads end a moment after ``close`` returns)."""
    until(lambda: not (sink_threads() - before), timeout)
    return sorted(sink_threads() - before, key=lambda t: t.name)
