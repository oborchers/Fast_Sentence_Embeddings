"""Tests of fse.enable_logging and fse.disable_logging, and of what fse's records do
without them (the stdlib's lastResort)."""

import collections
import io
import logging
import re
import subprocess
import sys
import threading

import pydantic
import pytest

import fse
from fse import _log, _log_config
from support.logs import TOP, suspended


class Err:
    """What the test wrote to sys.stderr (pytest's capture, which enable_logging takes)."""

    def __init__(self, capsys):
        self.capsys, self.text = capsys, ""

    def getvalue(self) -> str:
        self.text += self.capsys.readouterr().err
        return self.text


@pytest.fixture
def stderr(capsys):
    """sys.stderr of the test; fse's logging state restored after it, without the strict
    capture of conftest.py meanwhile (enable_logging refuses foreign handlers)."""
    level, propagate = TOP.level, TOP.propagate
    with suspended():
        try:
            yield Err(capsys)
        finally:
            fse.disable_logging()
            TOP.setLevel(level)
            TOP.propagate = propagate


def test_with_nothing_configured_a_warning_reaches_stderr():
    code = (
        "import fse\n"
        "from fse import _log\n"
        "_log.emit('runtime.arrow_copied', 'fse runtime: %s', 'slower', source='ArrowSource')\n"
        "_log.emit('store.opened', 'fse store: opened')\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stderr == "fse runtime: slower\n"  # INFO stays silent


@pytest.mark.parametrize(
    "setup",
    [
        "class Drop(logging.Filter):\n    def filter(self, record): return False\n"
        "h = logging.NullHandler(); h.addFilter(Drop()); logging.getLogger('fse').addHandler(h)",
        "logging.getLogger('fse.runtime').setLevel(logging.ERROR)",
        "logging.disable(logging.WARNING)",
    ],
    ids=["filtering-handler", "child-level", "disable"],
)
def test_last_resort_stays_silent(setup):
    code = f"import logging\nimport fse\nfrom fse import _log\n{setup}\n"
    code += "_log.emit('runtime.arrow_copied', 'fse runtime: slower', source='ArrowSource')\n"
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""


def test_last_resort_prints_without_propagation_and_handler():
    # CPython's callHandlers counts handlers, not the loggers it passed: none found.
    code = "import logging\nimport fse\nfrom fse import _log\n"
    code += "logging.getLogger('fse').propagate = False\n"
    code += "_log.emit('runtime.arrow_copied', 'fse runtime: slower', source='ArrowSource')\n"
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.stderr == "fse runtime: slower\n"


def _warning(message="fse runtime: slower"):
    _log.emit("runtime.arrow_copied", message, source="ArrowSource", reason="copy")


def test_enable_logging_prints_one_line_per_record(stderr):
    fse.enable_logging()
    _warning()
    _log.emit("store.opened", "fse store: opened", rows=3)
    _log.emit("store.search", "fse store: search")  # DEBUG, below INFO
    lines = stderr.getvalue().splitlines()
    assert len(lines) == 2
    assert re.fullmatch(r"\d\d:\d\d:\d\d\.\d{3} WARN  fse\.runtime  fse runtime: slower", lines[0])
    assert re.fullmatch(r"\d\d:\d\d:\d\d\.\d{3} INFO  fse\.store    fse store: opened", lines[1])
    assert TOP.level == logging.INFO
    assert TOP.propagate is False


def test_at_debug_lines_carry_the_fields_and_threads_their_names(stderr):
    fse.enable_logging("DEBUG")
    thread = threading.Thread(target=_warning, name="fse-worker_3")
    thread.start()
    thread.join()
    line = stderr.getvalue().strip()
    assert line.endswith(
        " [fse-worker_3] event=runtime.arrow_copied source=ArrowSource reason=copy"
    ), line


def test_colours_only_on_a_terminal_and_never_with_no_color(stderr, monkeypatch):
    terminal = io.StringIO()
    terminal.isatty = lambda: True
    monkeypatch.setattr(sys, "stderr", terminal)
    monkeypatch.delenv("NO_COLOR", raising=False)
    fse.enable_logging()
    _warning()
    assert "\033[33mWARN \033[0m" in terminal.getvalue()
    monkeypatch.setenv("NO_COLOR", "1")
    _warning()
    assert terminal.getvalue().splitlines()[1].split()[1] == "WARN"
    assert stderr.getvalue() == ""


def test_a_root_handler_prints_nothing_twice_and_child_levels_stay(stderr):
    root = io.StringIO()
    handler = logging.StreamHandler(root)
    logging.getLogger().addHandler(handler)
    store = logging.getLogger("fse.store")
    store.setLevel(logging.ERROR)
    try:
        fse.enable_logging()
        _warning()
        _log.emit("store.opened", "fse store: opened")  # below the level of fse.store
        assert len(stderr.getvalue().splitlines()) == 1
        assert root.getvalue() == ""
        fse.disable_logging()
        _warning("after")
        assert root.getvalue() == "after\n"
        assert store.level == logging.ERROR
    finally:
        logging.getLogger().removeHandler(handler)
        store.setLevel(logging.NOTSET)


@pytest.mark.parametrize("name", ["fse", "fse.store", "fse.store.deeper"])
def test_a_handler_of_the_application_makes_it_raise(stderr, name):
    logger = logging.getLogger(name)
    handler = logging.NullHandler()
    logger.addHandler(handler)
    try:
        with pytest.raises(RuntimeError, match=rf"NullHandler on '{re.escape(name)}'"):
            fse.enable_logging()
        assert TOP.propagate is True
        assert handler in logger.handlers
    finally:
        logger.removeHandler(handler)
    assert stderr.getvalue() == ""


@pytest.mark.usefixtures("stderr")
def test_enable_twice_reuses_the_handler_and_disable_restores():
    TOP.setLevel(logging.ERROR)
    fse.enable_logging()
    first = list(TOP.handlers)
    fse.enable_logging("DEBUG")
    assert len(first) == 1
    assert TOP.handlers == first
    assert TOP.level == logging.DEBUG
    fse.disable_logging()
    assert TOP.handlers == []
    assert (TOP.level, TOP.propagate) == (logging.ERROR, True)
    fse.disable_logging()  # nothing attached: nothing changes
    assert (TOP.level, TOP.propagate) == (logging.ERROR, True)
    fse.enable_logging()
    assert TOP.handlers == first  # the same handler, created once per process


@pytest.mark.usefixtures("stderr")
def test_values_the_application_changed_meanwhile_stay():
    fse.enable_logging()
    TOP.setLevel(logging.WARNING)
    TOP.propagate = True
    fse.disable_logging()
    assert (TOP.level, TOP.propagate) == (logging.WARNING, True)
    TOP.setLevel(logging.NOTSET)


def test_the_stream_is_the_stderr_of_each_call(stderr, monkeypatch):
    fse.enable_logging()
    later = io.StringIO()
    monkeypatch.setattr(sys, "stderr", later)
    fse.enable_logging()
    _warning()
    assert stderr.getvalue() == ""
    assert "slower" in later.getvalue()


@pytest.mark.parametrize("level", ["info", "CRITICAL", 20, None])
def test_the_level_is_validated(level):
    with pytest.raises(pydantic.ValidationError):
        fse.enable_logging(level)


def test_concurrent_records_while_it_attaches_and_detaches(stderr):
    root = io.StringIO()
    handler = logging.StreamHandler(root)
    logging.getLogger().addHandler(handler)
    stop = threading.Event()
    errors = []

    def emitting(worker):
        n = 0
        while not stop.is_set() and n < 3000:  # pytest's own root handlers keep each
            _warning(f"fse runtime: record {worker}-{n} end")
            n += 1

    def toggling():
        try:
            for i in range(1000):
                fse.enable_logging()
                _warning(f"fse runtime: record 99-{i} end")  # one while attached
                fse.disable_logging()
        except BaseException as error:
            errors.append(error)

    def levels():
        while not stop.is_set():
            TOP.setLevel(logging.DEBUG)
            TOP.setLevel(logging.NOTSET)

    threads = [threading.Thread(target=emitting, args=(i,)) for i in range(8)]
    threads += [threading.Thread(target=levels)]
    try:
        for thread in threads:
            thread.start()
        toggler = threading.Thread(target=toggling)
        toggler.start()
        toggler.join()
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        logging.getLogger().removeHandler(handler)
    assert errors == []
    lines = [*stderr.getvalue().splitlines(), *root.getvalue().splitlines()]
    records = [re.search(r"record (\d+-\d+) end", line) for line in lines]
    assert all(records), [line for line, r in zip(lines, records, strict=True) if not r][:3]
    # In flight while disable_logging restores propagation, a record may reach fse's
    # handler and then the root's (the stdlib reads propagate without a lock), never more.
    numbers = collections.Counter(r.group(1) for r in records)
    assert max(numbers.values()) <= 2
    printed = collections.Counter(re.findall(r"record (\d+-\d+) end", stderr.getvalue()))
    assert max(printed.values()) == 1
    assert _log_config._Owned.handler not in TOP.handlers


def test_a_broken_stream_is_reported_once(stderr, monkeypatch):
    class Broken(io.StringIO):
        def write(self, _text):
            raise OSError("the stream is gone")

    monkeypatch.setattr(_log, "_reported", set())
    report = io.StringIO()
    monkeypatch.setattr(sys, "__stderr__", report)  # where fse reports a failed record
    monkeypatch.setattr(sys, "stderr", Broken())
    fse.enable_logging()
    for _ in range(3):
        _warning()
    err = report.getvalue()
    assert err.count("logging the event runtime.arrow_copied failed with OSError") == 1
    assert "Traceback" not in err
    assert stderr.getvalue() == ""
