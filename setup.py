"""Build the Cython extension. All package metadata lives in pyproject.toml."""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "fse.models.average_inner",
        sources=["src/fse/models/average_inner.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
)
