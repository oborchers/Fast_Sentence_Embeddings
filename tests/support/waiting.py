"""Waits for a state that another thread reaches, bounded, for tests that must know a
thread arrived at a point before they act (a sleep only hopes it did)."""

import threading
import time


def blocked_in_wait(condition: threading.Condition, thread_name: str) -> threading.Event:
    """Make ``condition.wait`` of the thread named ``thread_name`` set the returned event
    before it blocks; ``arrived(event, condition)`` then proves the thread sits in it."""
    reached = threading.Event()
    real = condition.wait

    def wait(*args, **kwargs):
        if threading.current_thread().name == thread_name:
            reached.set()
        return real(*args, **kwargs)

    condition.wait = wait  # type: ignore[method-assign]
    return reached


def arrived(reached: threading.Event, condition: threading.Condition, timeout: float = 10) -> bool:
    """Whether the thread reached ``condition.wait`` and released the condition's lock in it
    (the lock can only be taken here once the thread waits)."""
    if not reached.wait(timeout):
        return False
    if not condition.acquire(timeout=timeout):
        return False
    condition.release()
    return True


def until(done, timeout: float = 10, step: float = 0.01) -> bool:
    """Whether ``done()`` turned true within ``timeout`` seconds (an event wait in a loop,
    never a sleep, and one monotone deadline for the whole wait)."""
    deadline = time.monotonic() + timeout
    pause = threading.Event()
    while not done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return done()
        pause.wait(min(step, remaining))
    return True


def announced(stream, predicate, timeout=10):
    """Read unbuffered JSON lines until a process announces the required state."""
    import json
    import os
    import selectors
    import time

    deadline = time.monotonic() + timeout
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise AssertionError(f"process did not announce state; partial output: {data!r}")
            chunk = os.read(stream.fileno(), 1)
            if not chunk:
                raise AssertionError(f"process reached EOF before announcement: {data!r}")
            data.extend(chunk)
            if chunk == b"\n":
                event = json.loads(data)
                if predicate(event):
                    return event
                data.clear()
