"""The records of fse in tests: the strict capture of conftest.py and ``expect_log``.

conftest.py attaches ``STRICT`` to the logger "fse" around the setup, call and teardown
of every test, so it sees records while propagation is off too; a WARNING or higher of
fse that no ``expect_log`` expected fails the test in its phase.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

from fse import _log

TOP = logging.getLogger("fse")


class Capture(logging.Handler):
    """Keeps every record it handles."""

    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


STRICT = Capture(logging.WARNING)
PAUSED: list[bool] = []  # non-empty while ``suspended``: conftest.py attaches no STRICT


def unexpected(records: list[logging.LogRecord]) -> list[str]:
    """The WARNING and higher records of ``records`` no ``expect_log`` expected."""
    return [
        f"{r.name} {r.levelname}: {r.getMessage()}"
        for r in records
        if r.levelno >= logging.WARNING and not getattr(r, "fse_expected", False)
    ]


@contextlib.contextmanager
def suspended() -> Iterator[None]:
    """The block, also across the phases of a test (a fixture), without ``STRICT`` on
    "fse" (``enable_logging`` refuses other handlers)."""
    attached = STRICT in TOP.handlers
    TOP.removeHandler(STRICT)
    PAUSED.append(True)
    try:
        yield
    finally:
        PAUSED.pop()
        if attached:
            TOP.addHandler(STRICT)


@contextlib.contextmanager
def captured(level: int = logging.DEBUG, *, mark: bool = True) -> Iterator[list[logging.LogRecord]]:
    """Every record of fse from ``level`` on within the block (each logger of fse set to
    ``level``), marked as expected unless ``mark`` is false."""
    capture = Capture(level)
    levels = {name: logging.getLogger(name).level for name in _log.LOGGER_NAMES}
    for name in _log.LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)
    TOP.addHandler(capture)
    try:
        yield capture.records
    finally:
        TOP.removeHandler(capture)
        for name, value in levels.items():
            logging.getLogger(name).setLevel(value)
        for record in capture.records if mark else ():
            record.fse_expected = True


@contextlib.contextmanager
def expect_log(event: str, **fields: object) -> Iterator[list[logging.LogRecord]]:
    """The records of ``event`` within the block (its level enabled meanwhile); at least
    one must have ``fields`` (``fse_<name>`` equal to each value)."""
    _, level = _log.EVENTS[event]
    with captured(level, mark=False) as records:
        mine: list[logging.LogRecord] = []
        yield mine
    mine += [r for r in records if r.fse_event == event]
    for record in mine:  # only these: another WARNING within the block still fails
        record.fse_expected = True
    assert any(
        all(getattr(r, f"fse_{k}", object()) == v for k, v in fields.items()) for r in mine
    ), f"no record of {event} with {fields}: {[vars(r) for r in mine]}"
