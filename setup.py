#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers
# For License information, see corresponding LICENSE file.

"""Template setup.py Read more on
https://docs.python.org/3.7/distutils/setupscript.html."""

import distutils
import itertools
import os
import platform
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

NAME = "fse"
VERSION = "0.1.16"
DESCRIPTION = "Fast Sentence Embeddings for Gensim"
AUTHOR = "Dr. Oliver Borchers"
AUTHOR_EMAIL = "borchers@bwl.uni-mannheim.de"
URL = "https://github.com/oborchers/Fast_Sentence_Embeddings"
LICENSE = "GPL-3.0"
REQUIRES_PYTHON = ">=3.8"
NUMPY_STR = "numpy >= 1.11.3"
CYTHON_STR = "Cython==0.29.14"

INSTALL_REQUIRES = [
    NUMPY_STR,
    "scipy >= 0.18.1",
    "smart_open >= 1.5.0",
    "scikit-learn >= 0.19.1",
    "gensim >= 3.8.0, < 4.0",
    "wordfreq >= 2.2.1",
    "psutil",
]
SETUP_REQUIRES = [NUMPY_STR]

c_extensions = {
    "fse.models.average_inner": "fse/models/average_inner.c",
}
cpp_extensions = {}


def need_cython():
    """Return True if we need Cython to translate any of the extensions.

    If the extensions have already been translated to C/C++, then we don"t need to
    install Cython and perform the translation.
    """
    expected = list(c_extensions.values()) + list(cpp_extensions.values())
    return any([not os.path.isfile(f) for f in expected])


def make_c_ext(use_cython=False):
    for module, source in c_extensions.items():
        if use_cython:
            source = source.replace(".c", ".pyx")
        extra_args = []
        #        extra_args.extend(["-g", "-O0"])  # uncomment if optimization limiting crash info
        yield Extension(
            module, sources=[source], language="c", extra_compile_args=extra_args,
        )


def make_cpp_ext(use_cython=False):
    extra_args = []
    system = platform.system()

    if system == "Linux":
        extra_args.append("-std=c++11")
    elif system == "Darwin":
        extra_args.extend(["-stdlib=libc++", "-std=c++11"])
    # extra_args.extend(["-g", "-O0"])  # uncomment if
    # optimization limiting crash info
    for module, source in cpp_extensions.items():
        if use_cython:
            source = source.replace(".cpp", ".pyx")
        yield Extension(
            module,
            sources=[source],
            language="c++",
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
        )


#
# We use use_cython=False here for two reasons:
#
# 1. Cython may not be available at this stage
# 2. The actual translation from Cython to C/C++ happens inside CustomBuildExt
#
ext_modules = list(
    itertools.chain(make_c_ext(use_cython=False), make_cpp_ext(use_cython=False))
)


class CustomBuildExt(build_ext):
    """Custom build_ext action with bootstrapping.

    We need this in order to use numpy and Cython in this script without importing them
    at module level, because they may not be available yet.
    """

    #
    # http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    #
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # https://docs.python.org/2/library/__builtin__.html#module-__builtin__
        __builtins__.__NUMPY_SETUP__ = False

        import numpy

        self.include_dirs.append(numpy.get_include())

        if need_cython():
            import Cython.Build

            Cython.Build.cythonize(list(make_c_ext(use_cython=True)))
            Cython.Build.cythonize(list(make_cpp_ext(use_cython=True)))


class CleanExt(distutils.cmd.Command):
    description = "Remove C sources, C++ sources and binaries for gensim extensions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for root, dirs, files in os.walk("gensim"):
            files = [
                os.path.join(root, f)
                for f in files
                if os.path.splitext(f)[1] in (".c", ".cpp", ".so")
            ]
            for f in files:
                self.announce("removing %s" % f, level=distutils.log.INFO)
                os.unlink(f)

        if os.path.isdir("build"):
            self.announce("recursively removing build", level=distutils.log.INFO)
            shutil.rmtree("build")


cmdclass = {"build_ext": CustomBuildExt, "clean_ext": CleanExt}

if need_cython():
    INSTALL_REQUIRES.append(CYTHON_STR)
    SETUP_REQUIRES.append(CYTHON_STR)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    requires_python=REQUIRES_PYTHON,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    include_package_data=True,
)
