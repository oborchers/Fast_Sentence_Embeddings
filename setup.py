"""Build the Cython extension. All package metadata lives in pyproject.toml."""

import os
import tempfile
from glob import glob

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# The kernel's rounding is part of the contract: no contraction of a multiply and an add
# other than the explicit ones, and no fast math (docs/architecture/numerical-contract.md,
# "Numerical contract (`transform_version` 2)").
FLAGS = {
    "unix": ["-ffp-contract=off", "-fno-fast-math"],
    "msvc": ["/fp:precise"],
}


# The NEON dot-product instruction (SDOT) of the int8 prefilter, built through a function
# attribute so the flags of the whole extension stay the same. The probe compiles one tiny
# function with it; a compiler that cannot builds the extension without that variant, which
# then does not exist in the binary (the CPU decides at run time whether it runs).
DOTPROD_PROBE = """
#if !defined(__aarch64__)
#error "SDOT is an aarch64 instruction"
#endif
#include <arm_neon.h>
#if defined(__clang__)
#define FSE_PROBE_TARGET __attribute__((target("dotprod")))
#else
#define FSE_PROBE_TARGET __attribute__((target("+dotprod")))
#endif
FSE_PROBE_TARGET int fse_probe(const signed char *a, const signed char *b) {
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b)));
}
"""


def compiles(compiler, source, flags):
    """Whether ``source`` compiles as C with ``flags`` (nothing is linked or kept)."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "probe.c")
        with open(path, "w") as handle:
            handle.write(source)
        try:
            compiler.compile([path], output_dir=directory, extra_postargs=flags)
        except Exception:
            return False
    return True


class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type not in FLAGS:
            # MinGW links fmaf from a runtime that rounds twice, which breaks the contract.
            raise RuntimeError(
                f"fse cannot be built with the {self.compiler.compiler_type} compiler; "
                "use MSVC on Windows, or GCC or Clang elsewhere"
            )
        flags = FLAGS[self.compiler.compiler_type]
        probe = self.compiler.compiler_type == "unix" and compiles(self.compiler, DOTPROD_PROBE, flags)
        for extension in self.extensions:
            extension.extra_compile_args = [*extension.extra_compile_args, *flags]
            if probe:
                extension.define_macros = [*extension.define_macros, ("FSE_HAVE_DOTPROD", "1")]
        super().build_extensions()


extensions = [
    Extension(
        "fse._native",
        sources=["src/fse/_native.pyx"],
        # The headers and the .pxi parts the .pyx includes: a change to any rebuilds it.
        depends=sorted(
            glob("src/fse/_native_src/**/*.h", recursive=True)
            + glob("src/fse/_native_src/**/*.pxi", recursive=True)
        ),
        include_dirs=[
            numpy.get_include(),
            "src/fse/_native_src",
            "src/fse/_native_src/kernel",
            "src/fse/_native_src/text",
            "src/fse/_native_src/tables",
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
    cmdclass={"build_ext": BuildExt},
)
