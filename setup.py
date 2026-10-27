"""Build the Cython extension. All package metadata lives in pyproject.toml."""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# The kernel's rounding is part of the contract: no contraction of a multiply and an add
# other than the explicit ones, and no fast math (ARCHITECTURE.md, "Numerical contract").
FLAGS = {
    "unix": ["-ffp-contract=off", "-fno-fast-math"],
    "mingw32": ["-ffp-contract=off", "-fno-fast-math"],
    "msvc": ["/fp:precise"],
}


class BuildExt(build_ext):
    def build_extensions(self):
        flags = FLAGS.get(self.compiler.compiler_type, [])
        for extension in self.extensions:
            extension.extra_compile_args = [*extension.extra_compile_args, *flags]
        super().build_extensions()


extensions = [
    Extension(
        "fse._native",
        sources=["src/fse/_native.pyx"],
        depends=["src/fse/_native_kernel.h"],
        include_dirs=[numpy.get_include(), "src/fse"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
    cmdclass={"build_ext": BuildExt},
)
