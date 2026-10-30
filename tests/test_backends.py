import pytest

from fse._backends import BACKEND_NAMES, NumpyBackend, get_backend
from support.logs import expect_log


def test_backends_by_name():
    assert "numpy" in BACKEND_NAMES
    assert isinstance(get_backend("numpy"), NumpyBackend)


@pytest.mark.parametrize("name", ["auto", "cuda", "", "NUMPY", "MLX"])
def test_unknown_backends_raise_with_the_list(name):
    with pytest.raises(ValueError, match="backend must be one of cpu, numpy, mlx"):
        get_backend(name)


def test_isa_belongs_to_the_cpu_backend():
    with pytest.raises(ValueError, match="cpu backend only"):
        get_backend("numpy", isa="portable")


def test_the_choice_of_a_backend_is_logged_once(monkeypatch):
    from fse import _backends

    monkeypatch.setattr(_backends, "_SELECTED", set())
    with expect_log("backend.selected", backend="numpy", isa=None, concurrent=True) as records:
        get_backend("numpy")
        get_backend("numpy")
        cpu = get_backend("cpu")
    assert [r.fse_backend for r in records] == ["numpy", "cpu"]
    assert records[1].fse_isa == cpu.isa
