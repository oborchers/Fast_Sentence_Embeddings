"""The package of linear heads."""

import fse._head


def test_the_package_imports_without_scikit_learn():
    assert fse._head.__doc__
