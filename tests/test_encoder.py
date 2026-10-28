import importlib


def test_the_encoder_package_imports():
    package = importlib.import_module("fse._encoder")
    assert package.__doc__
    assert importlib.import_module("fse._encoder._weights").sif_weights
