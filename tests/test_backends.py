import pytest

from fse._backends import BACKEND_NAMES, NumpyBackend, get_backend
from support.logs import expect_log


def test_backends_by_name():
    assert BACKEND_NAMES == ("cpu", "numpy")
    assert isinstance(get_backend("numpy"), NumpyBackend)


@pytest.mark.parametrize("name", ["auto", "cuda", "", "NUMPY", "MLX", "mlx"])
def test_unknown_backends_raise_with_the_list(name):
    with pytest.raises(ValueError, match="backend must be one of cpu, numpy"):
        get_backend(name)


def test_isa_belongs_to_the_cpu_backend():
    with pytest.raises(ValueError, match="cpu backend only"):
        get_backend("numpy", isa="portable")


def test_the_choice_of_a_backend_is_logged_once(monkeypatch):
    from fse import _backends

    monkeypatch.setattr(_backends, "_SELECTED", set())
    with expect_log("backend.selected", backend="numpy", isa=None) as records:
        get_backend("numpy")
        get_backend("numpy")
        cpu = get_backend("cpu")
    assert [r.fse_backend for r in records] == ["numpy", "cpu"]
    assert records[1].fse_isa == cpu.isa


@pytest.mark.parametrize("method", ["encode", "encode_batches"])
def test_removed_backend_is_rejected_before_reading_input(method):
    from support.encoder import finished

    def unread():
        yield pytest.fail("input was read before backend validation")

    result = getattr(finished(), method)

    def run():
        output = result(unread(), backend="mlx")
        if method == "encode_batches":
            next(output)

    with pytest.raises(ValueError, match=r"backend must be one of.*cpu.*numpy.*mlx"):
        run()
