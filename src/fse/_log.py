#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The records of fse, through the standard library only (ARCHITECTURE.md, "Logging").

fse logs on five fixed component loggers below ``fse``; their names, the events and the
fields of each event are fixed for 2.x. Every record carries ``fse_event`` and its fields
as ``fse_<name>`` attributes (``extra``), and none holds a path, an OS error text, a Hub
repository id or token, or input data. ``emit`` is the only way fse logs: it returns at
once for a level nobody listens to, and an exception of the application's handlers,
filters or record factory never reaches fse's code (it is reported once per event to
``sys.__stderr__``, as ``logging.Handler.handleError`` does). A field that costs work
(a hash, a stat) is computed only behind ``enabled`` at the call site, because the
arguments of ``emit`` are evaluated before it is called.

This module configures nothing: no handler, no level. ``fse.enable_logging`` does that on
request (``fse._log_config``).
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading

DEBUG, INFO, WARNING = logging.DEBUG, logging.INFO, logging.WARNING

#: The loggers of fse; "fse" is their parent and logs nothing itself.
LOGGER_NAMES = ("fse", "fse.artifact", "fse.hub", "fse.backend", "fse.runtime", "fse.store")

#: Every event of fse: its logger and level (the table in ARCHITECTURE.md, "Logging").
EVENTS: dict[str, tuple[str, int]] = {
    "backend.fallback": ("fse.backend", WARNING),
    "backend.fallbacks": ("fse.backend", DEBUG),
    "backend.selected": ("fse.backend", DEBUG),
    "backend.canary": ("fse.backend", DEBUG),
    "artifact.loaded": ("fse.artifact", INFO),
    "artifact.saved": ("fse.artifact", INFO),
    "artifact.verified": ("fse.artifact", INFO),
    "artifact.converted": ("fse.artifact", INFO),
    "artifact.ignored": ("fse.artifact", DEBUG),
    "artifact.cleanup_failed": ("fse.artifact", WARNING),
    "hub.resolved": ("fse.hub", INFO),
    "hub.file": ("fse.hub", DEBUG),
    "runtime.workers": ("fse.runtime", DEBUG),
    "runtime.plan": ("fse.runtime", DEBUG),
    "runtime.done": ("fse.runtime", DEBUG),
    "runtime.arrow_copied": ("fse.runtime", WARNING),
    "runtime.arrow_stream": ("fse.runtime", WARNING),
    "head.folded": ("fse.runtime", INFO),
    "store.sink_opened": ("fse.store", INFO),
    "store.shard_published": ("fse.store", DEBUG),
    "store.sink_closed": ("fse.store", INFO),
    "store.cleanup_failed": ("fse.store", WARNING),
    "store.opened": ("fse.store", INFO),
    "store.verified": ("fse.store", INFO),
    "store.search": ("fse.store", DEBUG),
}

_LOGGERS = {name: logging.getLogger(name) for name in LOGGER_NAMES}
_ROUTES = {event: (_LOGGERS[name], level) for event, (name, level) in EVENTS.items()}
_reported: set[str] = set()  # events whose logging failed, reported once each
_report_lock = threading.Lock()


def _route(event: str) -> tuple[logging.Logger, int]:
    try:
        return _ROUTES[event]
    except KeyError:
        raise ValueError(f"{event!r} is no event of fse._log.EVENTS") from None


def enabled(event: str) -> bool:
    """Whether a record of ``event`` would be handled at its level (``isEnabledFor``);
    False if the check itself fails (a logger class of the application)."""
    logger, level = _route(event)
    try:
        return logger.isEnabledFor(level)
    except Exception as error:
        _report(event, error)
        return False


def emit(event: str, message: str, *args: object, **fields: object) -> None:
    """Log ``message % args`` as ``event`` with ``fields`` (each as ``fse_<name>``).

    Returns at once when the level of the event is not enabled. Any ``Exception`` of the
    logging machinery (a handler, filter, formatter, record factory, or a field that
    cannot be formatted) is reported once per event and swallowed, so logging never
    changes a result or an exception of fse; KeyboardInterrupt and SystemExit propagate.
    """
    if not enabled(event):
        return
    logger, level = _ROUTES[event]
    extra: dict[str, object] = {"fse_event": event}
    for name, value in fields.items():
        if not name.isidentifier():
            raise ValueError(f"{name!r} is no identifier, so no field of an event")
        extra["fse_" + name] = value
    try:
        logger.log(level, message, *args, extra=extra, stacklevel=2)
    except Exception as error:
        _report(event, error)


def _report(event: str, error: Exception) -> None:
    """Say once per event on ``sys.__stderr__`` that logging it failed, without its fields
    (they would be formatted again), if ``logging.raiseExceptions`` is true."""
    with _report_lock:
        if event in _reported:
            return
        _reported.add(event)
    stream = sys.__stderr__
    if not logging.raiseExceptions or stream is None:
        return
    # A closed or broken stderr has nowhere left to report to.
    with contextlib.suppress(Exception):
        stream.write(
            f"fse: logging the event {event} failed with {type(error).__name__}; fse "
            "continues, and further failures of this event are not reported\n"
        )
