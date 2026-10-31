#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.enable_logging`` and ``fse.disable_logging``: fse's records on stderr, on request.

fse configures no logging by itself (docs/architecture/logging.md, "Logging"). These two functions are
the one exception: a handler owned by fse, created once and reused, on the logger "fse".
They never touch the root logger, the handlers of the application or the levels of the
loggers below "fse", and they restore only values nobody changed since.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Literal

from fse._log import LOGGER_NAMES, report_failure
from fse._validation import validate_arguments

_TOP = logging.getLogger("fse")
_LABELS = {logging.DEBUG: "DEBUG", logging.INFO: "INFO ", logging.WARNING: "WARN "}
_LABELS |= {logging.ERROR: "ERROR", logging.CRITICAL: "CRIT "}
_COLOURS = {logging.DEBUG: "2", logging.WARNING: "33", logging.ERROR: "31", logging.CRITICAL: "31"}


class _FseHandler(logging.StreamHandler):
    """The handler of ``enable_logging``: one line per record, on the stderr of the moment
    ``enable_logging`` was called."""

    def emit(self, record: logging.LogRecord) -> None:
        """StreamHandler.emit, but a record that cannot be formatted or written is
        reported once per event, as an error of the application's handlers is
        (fse._log), not with a traceback per record."""
        try:
            self.stream.write(self.format(record) + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception as error:
            report_failure(getattr(record, "fse_event", "fse"), error)

    def format(self, record: logging.LogRecord) -> str:
        clock = time.strftime("%H:%M:%S", time.localtime(record.created))
        label = _LABELS.get(record.levelno, f"{record.levelname:<5}"[:5])
        colour = _COLOURS.get(record.levelno)
        if colour is not None and _colours(self.stream):
            label = f"\033[{colour}m{label}\033[0m"
        line = f"{clock}.{int(record.msecs):03d} {label} {record.name:<12} {record.getMessage()}"
        if record.thread != threading.main_thread().ident:
            line += f" [{record.threadName}]"
        if _TOP.getEffectiveLevel() <= logging.DEBUG:
            fields = vars(record).items()
            line += "".join(f" {k[4:]}={v}" for k, v in fields if k.startswith("fse_"))
        return line


def _colours(stream: object) -> bool:
    """Colours only on a terminal, never with NO_COLOR set (https://no-color.org)."""
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return callable(isatty) and bool(isatty())


class _Owned:
    """What ``enable_logging`` changed on "fse", to restore it as it was."""

    handler: _FseHandler | None = None  # created once per process, then reused
    before: tuple[int, bool] | None = None  # level and propagate before the handler came
    level: int = logging.NOTSET  # the level enable_logging set


_lock = threading.Lock()


def _foreign_handlers() -> list[str]:
    """The handlers of the application on "fse" and the loggers below it."""
    found = []
    manager = logging.Logger.manager
    for name in sorted({*LOGGER_NAMES, *manager.loggerDict.copy()}):  # copy: atomic
        if name != "fse" and not name.startswith("fse."):
            continue
        logger = logging.getLogger(name)
        found += [
            f"{type(h).__name__} on {name!r}" for h in logger.handlers if h is not _Owned.handler
        ]
    return found


@validate_arguments
def enable_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO") -> None:
    """Print fse's records at ``level`` and above to stderr, one line each.

    Uses one handler owned by fse, created on the first call and reused afterwards, on the
    logger "fse"; sets the level of "fse" (not of the loggers below it) and stops the
    propagation from "fse" to the root logger while the handler is attached, so a handler
    of the root logger does not print the records a second time. At DEBUG every line ends
    with the fields of its record. Raises RuntimeError if the application attached a
    handler of its own to "fse" or a logger below it: logging of fse is configured then.
    """
    with _lock:
        foreign = _foreign_handlers()
        if foreign:
            raise RuntimeError(
                f"logging of fse is configured already ({', '.join(foreign)}); remove those "
                "handlers to use fse.enable_logging, or keep your own configuration"
            )
        if _Owned.handler is None:
            _Owned.handler = _FseHandler(sys.stderr)
        handler = _Owned.handler
        handler.acquire()  # not setStream, which flushes the old stream, perhaps closed
        try:
            handler.stream = sys.stderr
        finally:
            handler.release()
        numeric = logging.getLevelNamesMapping()[level]
        if handler not in _TOP.handlers:
            _Owned.before = (_TOP.level, _TOP.propagate)
            _TOP.propagate = False  # first: no record reaches both this handler and root
            _TOP.setLevel(numeric)
            _TOP.addHandler(handler)
        else:
            _TOP.setLevel(numeric)
        _Owned.level = numeric


def disable_logging() -> None:
    """Detach fse's handler and restore the level and propagation of "fse" to what they
    were before ``enable_logging``, each only if it still has the value ``enable_logging``
    set (a value the application changed since stays). Without ``enable_logging``,
    nothing changes. A record another thread emits at the same instant may still reach
    both fse's handler and a handler of the root logger: the standard library reads
    ``propagate`` after it called the handlers of "fse", without a lock."""
    with _lock:
        handler = _Owned.handler
        if handler is None or handler not in _TOP.handlers or _Owned.before is None:
            return
        _TOP.removeHandler(handler)  # first: no record printed twice once this returned
        level, propagate = _Owned.before
        if _TOP.level == _Owned.level:
            _TOP.setLevel(level)
        if _TOP.propagate is False:
            _TOP.propagate = propagate
        _Owned.before = None
