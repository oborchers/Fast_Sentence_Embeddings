"""Threads that read one store at once, for the tests of concurrent reads.

``Gate`` holds the first call of an internal step in each named reader thread until the
test opens it, and proves the reader sits there (``support.waiting``), so a test acts
while the reader holds its snapshot instead of hoping a sleep was long enough. ``Readers``
runs one callable per thread and raises the first error of a thread in the main thread.
``exact`` turns an outcome into its dtypes, shapes and bytes, so two outcomes compare bit
for bit whatever their types (float32 as its 32 bits, float16 as its 16, int8 codes,
float32 scales and int64 ids exactly).
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Mapping

import numpy as np

from support.waiting import arrived, blocked_in_wait

TIMEOUT = 60


class Gate:
    """Holds the first call of the steps given to ``hold`` in each of the threads ``names``
    until ``open``."""

    def __init__(self, names):
        self.condition = threading.Condition()
        self.opened = False
        self.held: set[str] = set()
        self.reached = {name: blocked_in_wait(self.condition, name) for name in names}

    def hold(self, monkeypatch, module, name: str) -> None:
        """Make ``module.name`` stop in an armed thread before it runs."""
        real = getattr(module, name)

        def held(*args, **kwargs):
            self.pause()
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, held)

    def pause(self) -> None:
        me = threading.current_thread().name
        with self.condition:
            if me not in self.reached or me in self.held:
                return
            self.held.add(me)
            assert self.condition.wait_for(lambda: self.opened, TIMEOUT), "the gate never opened"

    def all_arrived(self) -> bool:
        """Whether every armed thread waits in the gate."""
        return all(arrived(event, self.condition, TIMEOUT) for event in self.reached.values())

    def open(self) -> None:
        with self.condition:
            self.opened = True
            self.condition.notify_all()


class Readers:
    """One thread per callable of ``jobs`` (thread name to callable); ``join`` waits for all
    and raises the first error a thread raised, here in the main thread."""

    def __init__(self, jobs: Mapping[str, Callable[[], object]]):
        self.outcomes: dict[str, object] = {}
        self.errors: list[BaseException] = []
        self.threads = [
            threading.Thread(target=self._run, args=(name, job), name=name, daemon=True)
            for name, job in jobs.items()
        ]

    def _run(self, name: str, job: Callable[[], object]) -> None:
        try:
            self.outcomes[name] = job()
        except BaseException as error:  # raised again in the main thread by join
            self.errors.append(error)

    def start(self) -> Readers:
        for thread in self.threads:
            thread.start()
        return self

    def join(self) -> dict[str, object]:
        for thread in self.threads:
            thread.join(TIMEOUT)
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        assert not alive, f"threads did not finish: {alive}"
        if self.errors:
            raise self.errors[0]
        return self.outcomes


def exact(value):
    """The dtypes, shapes and bytes of an outcome (arrays, named tuples, dataclasses,
    mappings and lists of them), comparable with ``==`` bit for bit."""
    if isinstance(value, np.ndarray) and value.dtype.hasobject:  # labels: compare the objects
        return ("objects", value.shape, tuple(exact(item) for item in value.ravel().tolist()))
    if isinstance(value, np.ndarray):
        return ("array", str(value.dtype), value.shape, np.ascontiguousarray(value).tobytes())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return (type(value).__name__, tuple((f.name, exact(getattr(value, f.name))) for f in fields))
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, exact(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(exact(item) for item in value))
    return value
