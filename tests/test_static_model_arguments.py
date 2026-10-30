"""The arguments of StaticModel that change no row, and the refusal of the others."""

import numpy as np
import pytest

from fse._encoder._model2vec_rules import UNSET
from fse._static_model import _arguments as arguments

GOOD = {
    "show_progress_bar": False,
    "batch_size": 1024,
    "use_multiprocessing": True,
    "multiprocessing_threshold": 10_000,
    "convert_to_numpy": True,
    "convert_to_tensor": False,
    "device": None,
}


def test_the_neutral_values_give_the_workers_of_the_other_arguments():
    assert arguments.encode_arguments(**GOOD) is None
    assert arguments.encode_arguments(**{**GOOD, "use_multiprocessing": False}) == 1
    assert (
        arguments.encode_arguments(**{**GOOD, "device": "cpu", "show_progress_bar": True}) is None
    )


@pytest.mark.parametrize("name", ["show_progress_bar", "use_multiprocessing", "convert_to_numpy"])
@pytest.mark.parametrize("value", [1, 0, None, "True", np.bool_(True)])
def test_a_flag_is_a_bool(name, value):
    with pytest.raises(TypeError, match=f"{name} must be a bool"):
        arguments.encode_arguments(**{**GOOD, name: value})


@pytest.mark.parametrize("name", ["batch_size", "multiprocessing_threshold"])
@pytest.mark.parametrize("value", [True, 1.0, "3", None])
def test_a_count_is_an_int(name, value):
    with pytest.raises(TypeError, match=f"{name} must be an int"):
        arguments.encode_arguments(**{**GOOD, name: value})


def test_a_count_has_a_lower_bound():
    assert arguments.count("n", np.int16(3), 1) == 3
    assert type(arguments.count("n", np.int16(3), 1)) is int
    with pytest.raises(ValueError, match="at least 1"):
        arguments.count("n", 0, 1)
    assert arguments.count("n", 0, 0) == 0


def test_a_length_is_not_given_none_or_positive():
    assert arguments.length("max_length", UNSET) is UNSET
    assert arguments.length("max_length", None) is None
    assert arguments.length("max_length", np.int64(4)) == 4
    for value in (0, True, "4", 4.0):
        with pytest.raises((TypeError, ValueError)):
            arguments.length("max_length", value)


def test_loading_takes_float32_and_no_vocabulary_quantization():
    arguments.loading_arguments(None, None)
    arguments.loading_arguments("float32", None)
    for value in ("float16", "int8", "float64"):
        with pytest.raises(ValueError, match="model2vec computes the mean in that dtype"):
            arguments.loading_arguments(value, None)
    with pytest.raises(ValueError, match="dtypes of model2vec"):
        arguments.loading_arguments("Float32", None)
    with pytest.raises(TypeError, match="quantize_to must be a str"):
        arguments.loading_arguments(np.float32, None)
    with pytest.raises(ValueError, match="does not cluster"):
        arguments.loading_arguments(None, 8)
