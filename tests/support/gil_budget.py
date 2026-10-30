"""The guard of GIL hand-overs per task (tests/test_runtime_gil.py).

No API of CPython counts acquisitions of the GIL (ARCHITECTURE.md, "Runtime", Tasks), so
the guard counts the two ways the worker side of a task gives it up and takes it back:

- fse's own sections without the GIL, which every task-path entry point of fse._native
  counts right before it enters one (``_native._test_nogil_sections``), and which the
  source check ``uncounted`` keeps counted;
- numpy calls, which release the GIL above 500 elements: every function of the numpy
  module is wrapped (Cython looks a function up on the module at each call, so calls
  from the extension are counted too), and a profile function counts the ndarray
  methods and the blocking calls (os, time, locks) that Python code of the task makes.

``Budget`` runs the tasks of a pipeline one at a time, each in a window between its start
and its end, and records the counts of each window. ndarray operators and methods called
from Cython code are invisible to both counters; the source check forbids them in the
fused part (``forbidden``).
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import fse
from fse import _native
from fse._runtime import _pipeline

# Blocking calls a profile function sees as c_call: their modules or names.
_BLOCKING_MODULES = {"posix", "nt", "time", "_thread", "select"}
_BLOCKING_NAMES = {"acquire", "wait", "sleep"}


@dataclass
class Window:
    """The counts of one task."""

    sections: int = 0
    numpy: Counter = field(default_factory=Counter)
    profiled: Counter = field(default_factory=Counter)


class Budget:
    """While active: every task of every Pipeline runs alone, in a window whose counts are
    appended to ``windows``."""

    def __init__(self, monkeypatch):
        self.windows: list[Window] = []
        self._lock = threading.Lock()
        self._local = threading.local()
        self._wrap_numpy(monkeypatch)
        real = _pipeline.Pipeline._work
        budget = self

        def work(pipeline, local, task, target, base):
            with budget._lock:
                window = Window()
                budget._local.window = window
                before = _native._test_nogil_sections()
                try:
                    return real(pipeline, local, task, target, base)
                finally:
                    window.sections = _native._test_nogil_sections() - before
                    budget._local.window = None
                    budget.windows.append(window)

        monkeypatch.setattr(_pipeline.Pipeline, "_work", work)

    def _window(self) -> Window | None:
        return getattr(self._local, "window", None)

    def _wrap_numpy(self, monkeypatch):
        for name, value in list(vars(np).items()):  # no lazy submodule is imported
            if name.startswith("_") or inspect.isclass(value) or inspect.ismodule(value):
                continue
            if isinstance(value, np.ufunc) or not callable(value):
                continue
            monkeypatch.setattr(np, name, self._counting(name, value))

    def _counting(self, name, function):
        budget = self

        @functools.wraps(function)
        def counted(*args, **kwargs):
            window = budget._window()
            if window is not None:
                caller = sys._getframe(1).f_code.co_name
                window.numpy[f"{name} in {caller}"] += 1
            return function(*args, **kwargs)

        return counted

    def _profile(self, _frame, event, arg):
        window = self._window()
        if window is None or event != "c_call":
            return
        owner = getattr(arg, "__self__", None)
        module = getattr(arg, "__module__", None)
        name = getattr(arg, "__name__", "")
        if isinstance(owner, np.ndarray):
            window.profiled[f"ndarray.{name}"] += 1
        elif module in _BLOCKING_MODULES or name in _BLOCKING_NAMES:
            module = "os" if module in ("posix", "nt") else module  # the same on every OS
            window.profiled[f"{module}.{name}"] += 1

    def __enter__(self):
        threading.setprofile(self._profile)
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        threading.setprofile(None)  # type: ignore[arg-type]
        return False


# The task-path functions of the extension, by part: each of their sections without the
# GIL must be counted. Every section of _native_task.pxi must be.
TASK_PATH = {
    "_native_text.pxi": {"encode"},
    "_native_arrow_column.pxi": {"encode"},
    "_native_kernel.pxi": {"_run"},
    "_native_steps.pxi": {"_apply_steps"},
    "_native_store.pxi": {"store_rows"},
    "_native_search.pxi": {"_inverse_norms_into"},
}
FUSED = "_native_task.pxi"
_COUNT = "_nogil_sections += 1"
_DEF = re.compile(r"^(\s*)(?:c?p?def|cdef)\b.*?\b(\w+)\(")


def source(part: str) -> str:
    """The Cython source of a part of fse._native, as shipped with the package."""
    return (Path(fse.__file__).parent / part).read_text()


def _functions(text: str):
    """(name, first line, lines) of every def and cdef function of a Cython source: a body
    ends at the next line that is indented no more than its def."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = _DEF.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        end = next(
            (
                j
                for j in range(i + 1, len(lines))
                if lines[j].strip()
                and _indent(lines[j]) <= indent
                and not lines[j].strip().startswith(")")
            ),
            len(lines),
        )
        yield match.group(2), i, lines[i:end]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def uncounted(text: str, names: set[str] | None) -> list[str]:
    """The sections without the GIL of the functions ``names`` (all if None) that the
    line before them does not count."""
    found = []
    for name, first, body in _functions(text):
        if names is not None and name not in names:
            continue
        before = None
        for k, line in enumerate(body):
            counted = before is not None and before.strip() == _COUNT
            if line.strip().startswith("with nogil") and not counted:
                found.append(f"{name}, line {first + k + 1}")
            if line.strip():
                before = line
    return found


# What the fused part must not hold: a block or a function that takes the GIL back, and
# numpy calls or array methods (only the C API and the checks of types and flags).
_FORBIDDEN = [
    (re.compile(r"\bwith gil\b"), "with gil"),
    (re.compile(r"\bexcept \*"), "except *"),
    (
        re.compile(r"\bnp\.(?!PyArray_DATA\b|PyArray_NBYTES\b|ndarray\b|float32\b|integer\b)\w+"),
        "a numpy call",
    ),
    (re.compile(r"\.(any|all|max|min|sum|copy|view|astype|nonzero)\("), "an array method"),
]


def forbidden(text: str) -> list[str]:
    """The lines of ``text`` (code, not comments) with something the fused part must not
    hold."""
    found = []
    in_doc = False
    for k, line in enumerate(text.splitlines()):
        quotes = line.count('"""')
        if in_doc or quotes:
            in_doc = in_doc != (quotes % 2 == 1)
            continue  # a docstring, or a line that opens or closes one
        code = line.split("#", 1)[0]
        for pattern, what in _FORBIDDEN:
            if pattern.search(code):
                found.append(f"line {k + 1}: {what}")
    return found
