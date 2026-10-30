"""Tests of fse._log: the registry, enabled, emit and its error boundary."""

import logging
import re
from pathlib import Path

import pytest

from fse import _log
from support.logs import captured, expect_log

SRC = Path(_log.__file__).parent


def test_every_event_has_a_logger_of_fse_and_a_level():
    for event, (logger, level) in _log.EVENTS.items():
        assert logger in _log.LOGGER_NAMES[1:], event
        assert level in (logging.DEBUG, logging.INFO, logging.WARNING), event
        assert re.fullmatch(r"[a-z]+\.[a-z_]+", event), event


def test_import_configures_nothing():
    for name in _log.LOGGER_NAMES:
        logger = logging.getLogger(name)
        assert logger.level == logging.NOTSET, name
        assert logger.propagate, name
        assert not [h for h in logger.handlers if type(h).__module__.startswith("fse")], name


def test_unknown_events_and_fields_raise():
    with pytest.raises(ValueError, match="no event"):
        _log.emit("store.nope", "x")
    with pytest.raises(ValueError, match="no event"):
        _log.enabled("nope")
    with captured(), pytest.raises(ValueError, match="no identifier"):
        _log.emit("store.search", "x", **{"not a name": 1})


def test_emit_sets_the_event_and_prefixed_fields():
    with expect_log("store.search", queries=3, k=2) as records:
        _log.emit("store.search", "fse store: %d queries", 3, queries=3, k=2, name="clash")
    (record,) = records
    assert record.getMessage() == "fse store: 3 queries"
    assert record.name == "fse.store"
    assert record.levelno == logging.DEBUG
    assert record.fse_name == "clash"  # no KeyError of makeRecord for a reserved name
    assert record.funcName == "test_emit_sets_the_event_and_prefixed_fields"


def test_a_disabled_event_returns_at_once():
    assert not _log.enabled("store.search")  # DEBUG, and nothing configured
    seen = []

    class Loud:
        def __str__(self):
            seen.append(1)
            return "x"

    _log.emit("store.search", "%s", Loud())
    assert seen == []


class Raising(logging.Handler):
    def emit(self, _record):
        raise RuntimeError("the handler of the application failed")


class Refusing(logging.Filter):
    def filter(self, _record):
        raise RuntimeError("the filter of the application failed")


class Unprintable:
    def __str__(self):
        raise RuntimeError("no str")


def _fail_with(kind):
    logger = logging.getLogger("fse.store")
    if kind == "handler":
        return logger, "addHandler", "removeHandler", Raising()
    return logger, "addFilter", "removeFilter", Refusing()


@pytest.mark.parametrize("kind", ["handler", "filter"])
def test_the_boundary_swallows_and_reports_once(kind, capfd, monkeypatch):
    monkeypatch.setattr(_log, "_reported", set())  # each test sees the first failure
    logger, add, remove, part = _fail_with(kind)
    with captured():
        getattr(logger, add)(part)
        try:
            for _ in range(3):
                _log.emit("store.search", "x", queries=1)
        finally:
            getattr(logger, remove)(part)
    err = capfd.readouterr().err
    assert err.count("logging the event store.search failed with RuntimeError") == 1
    assert "queries" not in err


class Formatting(logging.Handler):
    def emit(self, record):
        self.format(record)  # unlike StreamHandler, no handleError of its own


def test_the_boundary_covers_record_factories_and_fields(capfd, monkeypatch):
    monkeypatch.setattr(_log, "_reported", set())
    previous = logging.getLogRecordFactory()

    def factory(*_args, **_kwargs):
        raise RuntimeError("the factory of the application failed")

    logger = logging.getLogger("fse.store")
    with captured():
        logging.setLogRecordFactory(factory)
        try:
            _log.emit("store.search", "x")
        finally:
            logging.setLogRecordFactory(previous)
        logger.addHandler(Formatting())
        try:
            _log.emit("store.opened", "fse store: %s", Unprintable())
        finally:
            logger.handlers.clear()
    err = capfd.readouterr().err
    assert "store.search failed with RuntimeError" in err
    assert "store.opened failed with RuntimeError" in err


def test_the_boundary_is_quiet_without_raise_exceptions(capfd, monkeypatch):
    monkeypatch.setattr(_log, "_reported", set())
    monkeypatch.setattr(logging, "raiseExceptions", False)
    logger = logging.getLogger("fse.store")
    with captured():
        logger.addHandler(Raising())
        try:
            _log.emit("store.search", "x")
        finally:
            logger.handlers.clear()
    assert capfd.readouterr().err == ""


def test_keyboard_interrupt_from_a_handler_propagates():
    class Interrupting(logging.Handler):
        def emit(self, _record):
            raise KeyboardInterrupt

    logger = logging.getLogger("fse.store")
    with captured():
        logger.addHandler(Interrupting())
        try:
            with pytest.raises(KeyboardInterrupt):
                _log.emit("store.search", "x")
        finally:
            logger.handlers.clear()
