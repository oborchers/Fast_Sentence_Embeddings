"""The guard of GIL hand-overs per task (tests/test_runtime_gil.py).

No API of CPython counts acquisitions of the GIL (docs/architecture/runtime.md, "Runtime", Tasks), so
the guard counts the two ways the worker side of a task gives it up and takes it back:

- fse's own sections without the GIL, which every task-path entry point of fse._native
  counts right before it enters one (``_native._test_nogil_sections``), and which the
  source check ``uncounted`` keeps counted;
- numpy calls, which release the GIL above 500 elements: every function and ufunc (with
  its methods) of the numpy module is wrapped (Cython looks a function up on the module at each call, so calls
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
            if isinstance(value, np.ufunc):
                monkeypatch.setattr(np, name, _Ufunc(self, name, value))
            elif callable(value):
                monkeypatch.setattr(np, name, self._counting(name, value))

    def count(self, name: str) -> None:
        """Count a numpy call of the caller's caller, inside a window."""
        window = self._window()
        if window is not None:
            window.numpy[f"{name} in {sys._getframe(2).f_code.co_name}"] += 1

    def _counting(self, name, function):
        budget = self

        @functools.wraps(function)
        def counted(*args, **kwargs):
            budget.count(name)
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


class _Ufunc:
    """A ufunc of the numpy module that counts its calls and those of its methods
    (``reduce``, ``accumulate``, ``at`` ...); every other attribute is the ufunc's.
    ``isinstance(np.add, np.ufunc)`` is False under it, which nothing in fse asks."""

    def __init__(self, budget: Budget, name: str, ufunc: np.ufunc):
        self._budget, self._name, self._ufunc = budget, name, ufunc

    def __call__(self, *args, **kwargs):
        self._budget.count(self._name)
        return self._ufunc(*args, **kwargs)

    def __getattr__(self, attr):
        value = getattr(self._ufunc, attr)
        if not callable(value):
            return value
        budget, name = self._budget, f"{self._name}.{attr}"

        def method(*args, **kwargs):
            budget.count(name)
            return value(*args, **kwargs)

        return method


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
    matches = [p for p in _native_parts() if p.name == part]
    assert len(matches) == 1, (part, matches)
    return matches[0].read_text()


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


# A cdef function called without the GIL must be noexcept: otherwise Cython checks for an
# exception after every call, and a void function takes the GIL back to do so (Cython
# docs, "Using the nogil statement").
_CALL = re.compile(r"(?:\b|\.)(\w+)\(")


def _cdef_signatures(texts: list[str]) -> dict[str, str]:
    """The signature (up to its colon) of every cdef function of the sources, by name."""
    found = {}
    for text in texts:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            match = re.match(r"^\s*cdef\b(?!\s+(?:class|struct|enum|extern)\b).*?\b(\w+)\(", line)
            if not match or "=" in line[: match.end()]:
                continue  # not a function, or a variable made from a call
            j = i
            while not lines[j].rstrip().endswith(":") and j + 1 < len(lines):
                j += 1
            found[match.group(1)] = " ".join(lines[i : j + 1])
    return found


def _nogil_calls(text: str) -> set[str]:
    """The names called inside the ``with nogil`` blocks of a source."""
    calls: set[str] = set()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().startswith("with nogil"):
            continue
        indent = _indent(line)
        for body in lines[i + 1 :]:
            if body.strip() and _indent(body) <= indent:
                break
            calls.update(_CALL.findall(body.split("#", 1)[0]))
    return calls


def checked_without_gil(fused: str, others: list[str]) -> list[str]:
    """The cdef functions the nogil blocks of ``fused`` call, directly or through each
    other, whose signature lacks ``noexcept``."""
    signatures = _cdef_signatures([fused, *others])
    bodies = {
        name: "\n".join(body) for text in (fused, *others) for name, _, body in _functions(text)
    }
    todo, seen, found = list(_nogil_calls(fused)), set(), []
    while todo:
        name = todo.pop()
        if name in seen or name not in signatures:
            continue  # a C function, or one already seen
        seen.add(name)
        if "noexcept" not in signatures[name]:
            found.append(name)
        body = bodies.get(name, "")
        todo.extend(_CALL.findall(body[body.find(":") + 1 :]))
    return sorted(found)


def native_sources() -> list[str]:
    """Every part of fse._native but the fused one."""
    return [p.read_text() for p in _native_parts() if p.name != FUSED]


def _native_parts() -> list[Path]:
    """Find every shipped part and verify it against the extension's include list."""
    folder = Path(fse.__file__).parent
    includes = re.findall(r'^include "([^"]+\.pxi)"', (folder / "_native.pyx").read_text(), re.M)
    parts = sorted(folder.rglob("_native*.pxi"))
    assert includes, "the extension includes no Cython parts"
    assert {p.relative_to(folder).as_posix() for p in parts} == set(includes)
    return parts
