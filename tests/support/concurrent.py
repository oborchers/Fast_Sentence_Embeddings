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

from fse._store import _rescore, _search, _selection, _take, _vectorstore
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

    def hold(self, monkeypatch, module, name: str, threads=None, *, after=False) -> None:
        """Make ``module.name`` stop in an armed thread (of ``threads``, default all) before
        it runs, or with ``after`` once it has returned."""
        real = getattr(module, name)
        chosen = set(self.reached if threads is None else threads)

        def held(*args, **kwargs):
            mine = threading.current_thread().name in chosen
            if mine and not after:
                self.pause()
            result = real(*args, **kwargs)
            if mine and after:
                self.pause()
            return result

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

    def wait(self) -> None:
        """Wait for every thread that was started, without raising its error (for the clean-up
        of a test that failed meanwhile)."""
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(TIMEOUT)

    def join(self) -> dict[str, object]:
        self.wait()
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        assert not alive, f"threads did not finish: {alive}"
        if self.errors:
            raise self.errors[0]
        return self.outcomes


def repeated(read, target, times, start=None):
    """A job that calls ``read(target)`` ``times`` times (after the barrier ``start``, if
    given) and returns the exact outcomes."""

    def run():
        if start is not None:
            start.wait()
        return [exact(read(target)) for _ in range(times)]

    return run


def exact(value):
    """The dtypes, shapes and bytes of an outcome (arrays, named tuples, dataclasses,
    mappings and lists of them), comparable with ``==`` bit for bit."""
    if isinstance(value, np.ndarray):
        return _array(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return (
            type(value).__name__,
            tuple((f.name, exact(getattr(value, f.name))) for f in fields),
        )
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, exact(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(exact(item) for item in value))
    return value


def _array(value):
    if value.dtype.hasobject:  # labels: compare the objects, not their addresses
        return ("objects", value.shape, tuple(exact(item) for item in value.ravel().tolist()))
    return ("array", str(value.dtype), value.shape, np.ascontiguousarray(value).tobytes())


def steps_of(name):
    """The steps a read takes under the active-use guard of the store, where a gate holds it
    (``rescore`` also calls ``_selection.plan``, before the guard)."""
    if name.startswith("search"):
        return [(_selection, "plan")]
    if name.startswith("take"):
        return [(_take, "gather")]
    if name.startswith("score"):
        return [(_search, "score")]
    if name == "rescore":
        return [(_rescore, "check_values")]
    if name == "verify":  # a store checks each shard, a view verifies its stores
        return [(_vectorstore, "check_removed"), (_vectorstore, "verify_members")]
    return [(_vectorstore, "stop_row")]  # get_* and iter_batches


def hold_every_read(gate, monkeypatch, reads):
    """Hold each read of ``reads`` (named as ``steps_of`` expects) at its guarded step."""
    steps: dict = {}
    for name in reads:
        for step in steps_of(name):
            steps.setdefault(step, []).append(name)
    for (module, step), names in steps.items():
        gate.hold(monkeypatch, module, step, threads=names)
