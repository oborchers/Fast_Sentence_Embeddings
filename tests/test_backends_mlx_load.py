"""fse._backends._mlx._load: the backend is refused with the reason it cannot run.

These tests replace the platform and MLX with fakes; they need mlx installed only
because the loader imports it (tests/test_backends_mlx.py runs without it).
"""

import types

import pytest

from fse._backends import UnsupportedByBackendError
from support.conformance import mlx_module

_load = mlx_module("fse._backends._mlx._load")


@pytest.fixture
def probe(monkeypatch):
    """A fresh probe on Apple silicon with macOS 15 and a working MLX, each part replaceable."""
    metal = types.SimpleNamespace(is_available=lambda: True)
    fake = types.SimpleNamespace(__version__="0.32.4", metal=metal)
    monkeypatch.setattr(_load.sys, "platform", "darwin")
    monkeypatch.setattr(_load.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(_load.platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64"))
    monkeypatch.setattr(_load, "mx", fake)
    _load._probe.cache_clear()
    yield fake
    _load._probe.cache_clear()


def test_core_gives_mlx_where_it_runs(probe):
    assert _load.core() is probe


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda m, _f: m.setattr(_load.sys, "platform", "linux"), "this is linux on arm64"),
        (lambda m, _f: m.setattr(_load.platform, "machine", lambda: "x86_64"), "on x86_64"),
        (
            lambda m, _f: m.setattr(_load.platform, "mac_ver", lambda: ("13.6.1", (), "")),
            "this is macOS 13.6.1",
        ),
        (lambda m, f: m.setattr(f, "__version__", "0.32.3"), "mlx 0.32.3 is installed, 0.32.4"),
        (lambda m, f: m.setattr(f.metal, "is_available", lambda: False), "Metal is not available"),
    ],
    ids=["linux", "intel", "macos-13", "old-mlx", "no-metal"],
)
def test_core_names_what_is_missing(probe, monkeypatch, change, reason):
    change(monkeypatch, probe)
    with pytest.raises(UnsupportedByBackendError, match=r'pip install "fse\[mlx\]"') as err:
        _load.core()
    assert reason in str(err.value)


def test_the_probe_runs_once(probe, monkeypatch):
    calls = []
    monkeypatch.setattr(probe.metal, "is_available", lambda: calls.append(1) or True)
    _load.core()
    _load.core()
    assert calls == [1]


@pytest.mark.parametrize(
    ("text", "version"),
    [("0.32.4", (0, 32, 4)), ("0.32.4.dev20261028", (0, 32, 4)), ("1.0rc1", (1, 0)), ("", ())],
)
def test_versions_compare_by_their_numbers(text, version):
    assert _load._version(text) == version
