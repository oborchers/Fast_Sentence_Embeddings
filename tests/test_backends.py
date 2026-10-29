import pytest

from fse._backends import BACKEND_NAMES, NumpyBackend, get_backend


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
