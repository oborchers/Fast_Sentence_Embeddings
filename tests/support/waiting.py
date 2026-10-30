"""Waits for a state that another thread reaches, bounded, for tests that must know a
thread arrived at a point before they act (a sleep only hopes it did)."""

import threading
import time
from collections.abc import Callable


def until(predicate: Callable[[], object], timeout: float = 10) -> bool:
    """Whether ``predicate()`` became true within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


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
